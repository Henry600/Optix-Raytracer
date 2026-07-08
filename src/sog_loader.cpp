// sog_loader.cpp — SOG v2 (Spatially Ordered Gaussians) gaussian splat loader.
//
// Format reference:
//   https://developer.playcanvas.com/user-manual/gaussian-splatting/formats/sog/
//
// A SOG dataset is a meta.json plus a set of lossless WebP property images,
// either as loose files next to the meta.json or bundled into a single ZIP
// archive (.sog) with all files at the archive root.  Per-gaussian properties
// are co-located: linear pixel index i across every property image belongs to
// gaussian i.
//
// Decode summary (all resolved at load time into flat float arrays):
//   positions — 16-bit/axis split across means_l + means_u, normalized into
//               per-axis [mins, maxs] ranges, then inverse symmetric-log:
//               sign(n) * (exp(|n|) - 1)
//   rotations — "smallest three" quaternion: RGB store three components
//               quantized to [-√2/2, +√2/2], alpha 252..255 selects which of
//               (w,x,y,z) was omitted and is rebuilt from the unit constraint
//   scales    — 8-bit indices into a 256-entry log-domain codebook, exp()
//               recovers linear scale
//   sh0       — 8-bit indices into a codebook of DC SH coefficients (f_dc);
//               alpha is opacity / 255
//   shN       — optional palette: labels holds a 16-bit palette index per
//               gaussian (R + G<<8), centroids holds codebook-indexed RGB
//               coefficient triples, 64 palette entries per texture row

#include "sog_loader.h"

#include "json.hpp"  // nlohmann/json, bundled with tinygltf
#include "miniz.h"
#include <webp/decode.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <vector>

namespace
{

using json = nlohmann::json;

// ─── File access ──────────────────────────────────────────────────────────────

bool readWholeFile(const std::filesystem::path& p, std::vector<uint8_t>& out, std::string& error)
{
    std::ifstream f(p, std::ios::binary | std::ios::ate);
    if (!f)
    {
        error = "cannot open file: " + p.u8string();
        return false;
    }
    const std::streamsize size = f.tellg();
    f.seekg(0, std::ios::beg);
    out.resize(static_cast<size_t>(size));
    if (!f.read(reinterpret_cast<char*>(out.data()), size))
    {
        error = "read failed: " + p.u8string();
        return false;
    }
    return true;
}

// Abstracts "get file by name" over the two SOG layouts (ZIP bundle / directory).
class FileProvider
{
public:
    virtual ~FileProvider() = default;
    virtual bool read(const std::string& name, std::vector<uint8_t>& out, std::string& error) = 0;
};

class DirProvider : public FileProvider
{
public:
    explicit DirProvider(std::filesystem::path dir) : m_dir(std::move(dir)) {}

    bool read(const std::string& name, std::vector<uint8_t>& out, std::string& error) override
    {
        return readWholeFile(m_dir / std::filesystem::u8path(name), out, error);
    }

private:
    std::filesystem::path m_dir;
};

class ZipProvider : public FileProvider
{
public:
    ~ZipProvider() override
    {
        if (m_open)
        {
            mz_zip_reader_end(&m_zip);
        }
    }

    bool init(const std::filesystem::path& p, std::string& error)
    {
        if (!readWholeFile(p, m_bytes, error))
        {
            return false;
        }
        std::memset(&m_zip, 0, sizeof(m_zip));
        if (!mz_zip_reader_init_mem(&m_zip, m_bytes.data(), m_bytes.size(), 0))
        {
            error = "not a valid ZIP archive: " + p.u8string();
            return false;
        }
        m_open = true;
        return true;
    }

    bool read(const std::string& name, std::vector<uint8_t>& out, std::string& error) override
    {
        const int idx = mz_zip_reader_locate_file(&m_zip, name.c_str(), nullptr, 0);
        if (idx < 0)
        {
            error = "'" + name + "' not found in SOG archive";
            return false;
        }
        mz_zip_archive_file_stat st = {};
        if (!mz_zip_reader_file_stat(&m_zip, static_cast<mz_uint>(idx), &st))
        {
            error = "cannot stat '" + name + "' in SOG archive";
            return false;
        }
        out.resize(static_cast<size_t>(st.m_uncomp_size));
        if (!mz_zip_reader_extract_to_mem(&m_zip, static_cast<mz_uint>(idx), out.data(), out.size(), 0))
        {
            error = "failed to extract '" + name + "' from SOG archive";
            return false;
        }
        return true;
    }

private:
    std::vector<uint8_t> m_bytes;   // archive kept in memory for the reader's lifetime
    mz_zip_archive       m_zip  = {};
    bool                 m_open = false;
};

// ─── WebP decoding ────────────────────────────────────────────────────────────

// RAII wrapper around WebPDecodeRGBA output.  Decoding always requests RGBA so
// RGB-only images arrive with alpha = 255 and every image indexes uniformly.
struct DecodedImage
{
    uint8_t* rgba   = nullptr;
    int      width  = 0;
    int      height = 0;

    DecodedImage() = default;
    DecodedImage(const DecodedImage&)            = delete;
    DecodedImage& operator=(const DecodedImage&) = delete;

    ~DecodedImage()
    {
        if (rgba)
        {
            WebPFree(rgba);
        }
    }

    uint64_t pixelCount() const
    {
        return static_cast<uint64_t>(width) * static_cast<uint64_t>(height);
    }
};

bool decodeWebp(FileProvider& files, const std::string& name, uint64_t minPixels,
                DecodedImage& img, std::string& error)
{
    std::vector<uint8_t> bytes;
    if (!files.read(name, bytes, error))
    {
        return false;
    }
    img.rgba = WebPDecodeRGBA(bytes.data(), bytes.size(), &img.width, &img.height);
    if (!img.rgba)
    {
        error = "WebP decode failed for '" + name + "'";
        return false;
    }
    if (img.pixelCount() < minPixels)
    {
        error = "'" + name + "' is too small for the gaussian count";
        return false;
    }
    return true;
}

// ─── meta.json helpers ────────────────────────────────────────────────────────

std::string sectionFile(const json& section, size_t idx)
{
    return section.at("files").at(idx).get<std::string>();
}

std::vector<float> readCodebook(const json& section)
{
    std::vector<float> cb = section.at("codebook").get<std::vector<float>>();
    if (cb.empty())
    {
        throw std::runtime_error("empty codebook in meta.json");
    }
    return cb;
}

// Codebook indices are bytes; clamp against short codebooks rather than read OOB.
inline float cbLookup(const std::vector<float>& cb, uint8_t idx)
{
    return cb[std::min<size_t>(idx, cb.size() - 1)];
}

// Inverse of the symmetric log transform applied to positions at encode time.
inline float unlog(float n)
{
    const float s = (n >= 0.0f) ? 1.0f : -1.0f;
    return s * (std::exp(std::fabs(n)) - 1.0f);
}

} // anonymous namespace

// ─── loadSog ──────────────────────────────────────────────────────────────────

bool loadSog(const std::string& path, GaussianSplatData& out, std::string& error)
{
    try
    {
        const std::filesystem::path fsPath = std::filesystem::u8path(path);

        // ── Pick layout: ZIP bundle (.sog) or loose files next to meta.json ──
        std::unique_ptr<FileProvider> files;
        std::vector<uint8_t>          metaBytes;

        std::string ext = fsPath.extension().u8string();
        std::transform(ext.begin(), ext.end(), ext.begin(),
                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });

        if (ext == ".json")
        {
            files = std::make_unique<DirProvider>(fsPath.parent_path());
            if (!readWholeFile(fsPath, metaBytes, error))
            {
                return false;
            }
        }
        else
        {
            auto zip = std::make_unique<ZipProvider>();
            if (!zip->init(fsPath, error))
            {
                return false;
            }
            files = std::move(zip);
            if (!files->read("meta.json", metaBytes, error))
            {
                return false;
            }
        }

        const json meta = json::parse(metaBytes.begin(), metaBytes.end());

        const int version = meta.at("version").get<int>();
        if (version != 2)
        {
            error = "unsupported SOG version " + std::to_string(version)
                  + " (only version 2 is supported)";
            return false;
        }

        const uint32_t count = meta.at("count").get<uint32_t>();
        if (count == 0)
        {
            error = "SOG contains no gaussians";
            return false;
        }

        // ── Decode the required property images ──────────────────────────────
        const json& jMeans  = meta.at("means");
        const json& jScales = meta.at("scales");
        const json& jQuats  = meta.at("quats");
        const json& jSh0    = meta.at("sh0");

        DecodedImage meansL, meansU, quats, scales, sh0;
        if (!decodeWebp(*files, sectionFile(jMeans, 0), count, meansL, error) ||
            !decodeWebp(*files, sectionFile(jMeans, 1), count, meansU, error) ||
            !decodeWebp(*files, sectionFile(jQuats, 0), count, quats,  error) ||
            !decodeWebp(*files, sectionFile(jScales, 0), count, scales, error) ||
            !decodeWebp(*files, sectionFile(jSh0, 0), count, sh0,    error))
        {
            return false;
        }

        float mins[3];
        float maxs[3];
        for (int a = 0; a < 3; ++a)
        {
            mins[a] = jMeans.at("mins").at(a).get<float>();
            maxs[a] = jMeans.at("maxs").at(a).get<float>();
        }

        const std::vector<float> scaleCb = readCodebook(jScales);
        const std::vector<float> sh0Cb   = readCodebook(jSh0);

        // ── Dequantize per-gaussian properties ────────────────────────────────
        out = GaussianSplatData{};
        out.count     = count;
        out.antialias = meta.value("antialias", false);
        out.name      = fsPath.stem().u8string();
        out.positions.resize(count);
        out.rotations.resize(count);
        out.scales.resize(count);
        out.sh0.resize(count);
        out.opacities.resize(count);

        constexpr float kQuatRange = 1.41421356237f;  // 2 / sqrt(2)

        for (uint32_t i = 0; i < count; ++i)
        {
            const size_t px = static_cast<size_t>(i) * 4;

            // Position: 16-bit per axis → per-axis range lerp → inverse log
            float p[3];
            for (int a = 0; a < 3; ++a)
            {
                const uint32_t q = static_cast<uint32_t>(meansL.rgba[px + a])
                                 | (static_cast<uint32_t>(meansU.rgba[px + a]) << 8);
                const float n = mins[a] + (maxs[a] - mins[a]) * (static_cast<float>(q) / 65535.0f);
                p[a] = unlog(n);
            }
            out.positions[i] = make_float3(p[0], p[1], p[2]);

            // Rotation: smallest-three, alpha mode selects the omitted component
            {
                const uint8_t modeByte = quats.rgba[px + 3];
                if (modeByte < 252)
                {
                    error = "invalid quaternion mode byte at gaussian " + std::to_string(i);
                    return false;
                }
                const int mode = modeByte - 252;  // 0=w, 1=x, 2=y, 3=z omitted

                const float stored[3] = {
                    (static_cast<float>(quats.rgba[px + 0]) / 255.0f - 0.5f) * kQuatRange,
                    (static_cast<float>(quats.rgba[px + 1]) / 255.0f - 0.5f) * kQuatRange,
                    (static_cast<float>(quats.rgba[px + 2]) / 255.0f - 0.5f) * kQuatRange,
                };
                const float d = std::sqrt(std::max(0.0f,
                    1.0f - stored[0] * stored[0] - stored[1] * stored[1] - stored[2] * stored[2]));

                float wxyz[4];
                int j = 0;
                for (int k = 0; k < 4; ++k)
                {
                    if (k == mode)
                    {
                        continue;
                    }
                    wxyz[k] = stored[j];
                    ++j;
                }
                wxyz[mode] = d;

                out.rotations[i] = make_float4(wxyz[1], wxyz[2], wxyz[3], wxyz[0]);  // (x,y,z,w)
            }

            // Scale: log-domain codebook → exp → linear
            out.scales[i] = make_float3(
                std::exp(cbLookup(scaleCb, scales.rgba[px + 0])),
                std::exp(cbLookup(scaleCb, scales.rgba[px + 1])),
                std::exp(cbLookup(scaleCb, scales.rgba[px + 2])));

            // DC color coefficients (f_dc) + opacity
            out.sh0[i] = make_float3(
                cbLookup(sh0Cb, sh0.rgba[px + 0]),
                cbLookup(sh0Cb, sh0.rgba[px + 1]),
                cbLookup(sh0Cb, sh0.rgba[px + 2]));
            out.opacities[i] = static_cast<float>(sh0.rgba[px + 3]) / 255.0f;
        }

        // ── Optional higher-order spherical harmonics ─────────────────────────
        if (meta.contains("shN") && !meta.at("shN").is_null())
        {
            const json& jShN = meta.at("shN");

            const int bands = jShN.at("bands").get<int>();
            if (bands < 1 || bands > 3)
            {
                error = "invalid shN band count " + std::to_string(bands);
                return false;
            }
            const uint32_t paletteCount = jShN.at("count").get<uint32_t>();
            const std::vector<float> shNCb = readCodebook(jShN);

            DecodedImage centroids, labels;
            // Centroids: palette entries packed row-major, coeffsPerChannel
            // pixels per entry.  The spec uses 64 entries per row (width =
            // 64 * coeffs); deriving the count from the actual width handles
            // that and stays correct for any conforming multiple.
            const int coeffs = (bands + 1) * (bands + 1) - 1;  // 3 / 8 / 15
            if (!decodeWebp(*files, sectionFile(jShN, 0), 1, centroids, error) ||
                !decodeWebp(*files, sectionFile(jShN, 1), count, labels, error))
            {
                return false;
            }
            const uint32_t entriesPerRow = static_cast<uint32_t>(centroids.width) / coeffs;
            if (entriesPerRow == 0)
            {
                error = "shN centroids image narrower than one palette entry";
                return false;
            }

            out.shBands = bands;
            out.shN.resize(static_cast<size_t>(count) * coeffs * 3);

            for (uint32_t i = 0; i < count; ++i)
            {
                const size_t   lpx = static_cast<size_t>(i) * 4;
                const uint32_t n   = static_cast<uint32_t>(labels.rgba[lpx + 0])
                                   | (static_cast<uint32_t>(labels.rgba[lpx + 1]) << 8);
                if (n >= paletteCount)
                {
                    error = "shN palette index out of range at gaussian " + std::to_string(i);
                    return false;
                }

                const uint32_t v     = n / entriesPerRow;
                const uint32_t uBase = (n % entriesPerRow) * static_cast<uint32_t>(coeffs);
                if (v >= static_cast<uint32_t>(centroids.height))
                {
                    error = "shN centroid row out of range at gaussian " + std::to_string(i);
                    return false;
                }
                float* dst = out.shN.data() + static_cast<size_t>(i) * coeffs * 3;

                for (int c = 0; c < coeffs; ++c)
                {
                    const size_t cpx =
                        (static_cast<size_t>(v) * centroids.width + uBase + c) * 4;
                    dst[c * 3 + 0] = cbLookup(shNCb, centroids.rgba[cpx + 0]);
                    dst[c * 3 + 1] = cbLookup(shNCb, centroids.rgba[cpx + 1]);
                    dst[c * 3 + 2] = cbLookup(shNCb, centroids.rgba[cpx + 2]);
                }
            }
        }

        return true;
    }
    catch (const std::exception& e)
    {
        error = std::string("SOG parse error: ") + e.what();
        return false;
    }
}
