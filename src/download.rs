//! 模型自动下载与解包，对应 CnocrCpp 的 `model_files.cpp`。
//!
//! 模型托管在 `breezedeus/cnstd-cnocr-models`，每个模型是一个 `<name>-onnx.zip`，包里只有一个
//! `.onnx`（可能带子目录前缀）。下载用 `ureq`（TLS 沿用 ort 的 native-tls），解包只需要 zip
//! 中央目录 + raw deflate，所以只依赖已经在依赖树里的 `flate2`。

use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::time::Duration;

use flate2::bufread::DeflateDecoder;

/// 模型仓库里 `resolve/main/` 的路径前缀
const REPO_PREFIX: &str = "breezedeus/cnstd-cnocr-models/resolve/main/";
/// 单次下载（含重定向与读 body）的总超时
const DOWNLOAD_TIMEOUT: Duration = Duration::from_secs(180);
/// 中央目录结尾记录（EOCD）的最长回溯距离：22 字节固定部分 + 65535 字节注释
const MAX_EOCD_SEARCH: usize = 65557;
/// 中央目录/本地文件头签名
const CENTRAL_SIGNATURE: u32 = 0x0201_4b50;
const LOCAL_SIGNATURE: u32 = 0x0403_4b50;
const EOCD_SIGNATURE: u32 = 0x0605_4b50;
/// ZIP64 的 32 位字段占位值
const ZIP64_MARKER: u32 = u32::MAX;

fn le16(bytes: &[u8], at: usize) -> u16 {
    u16::from_le_bytes(bytes[at..at + 2].try_into().expect("zip 头部长度不足"))
}

fn le32(bytes: &[u8], at: usize) -> u32 {
    u32::from_le_bytes(bytes[at..at + 4].try_into().expect("zip 头部长度不足"))
}

/// 去掉末尾多余的 `/`
fn trim_slash(endpoint: &str) -> &str {
    endpoint.trim_end_matches('/')
}

/// 依次尝试的下载地址：`CNOCR_MODEL_ENDPOINT` 非空时只用它；否则 `HF_ENDPOINT`（若有）排在
/// 两个公共镜像前面。
fn endpoints(cnocr: Option<&str>, hf: Option<&str>) -> Vec<String> {
    let mut list: Vec<String> = Vec::new();
    let mut push = |value: &str| {
        let value = trim_slash(value).to_string();
        if !value.is_empty() && !list.contains(&value) {
            list.push(value);
        }
    };
    // 只有非空的 CNOCR_MODEL_ENDPOINT 才覆盖默认列表
    if let Some(value) = cnocr.map(trim_slash).filter(|value| !value.is_empty()) {
        return vec![value.to_string()];
    }
    if let Some(value) = hf {
        push(value);
    }
    push("https://hf-mirror.com");
    push("https://huggingface.co");
    list
}

/// 读取环境变量后的下载地址列表
pub fn download_endpoints() -> Vec<String> {
    let cnocr = std::env::var("CNOCR_MODEL_ENDPOINT").ok();
    let hf = std::env::var("HF_ENDPOINT").ok();
    endpoints(cnocr.as_deref(), hf.as_deref())
}

/// 拼出压缩包地址，例如
/// `https://hf-mirror.com/breezedeus/cnstd-cnocr-models/resolve/main/models/cnstd/1.2/ch_PP-OCRv3_det_infer-onnx.zip`
pub fn archive_url(endpoint: &str, subfolder: &str, archive: &str) -> String {
    let mut url = trim_slash(endpoint).to_string();
    url.push('/');
    url.push_str(REPO_PREFIX);
    if !subfolder.is_empty() {
        url.push_str(trim_slash(subfolder));
        url.push('/');
    }
    url.push_str(archive);
    url
}

/// ureq 默认用 rustls，这里显式切到 native-tls（Windows 走系统 schannel），
/// 否则会命中 “feature is not enabled: rustls” 并触发 C 编译的 ring
fn http_agent() -> ureq::Agent {
    let tls = ureq::tls::TlsConfig::builder().provider(ureq::tls::TlsProvider::NativeTls).build();
    let config = ureq::Agent::config_builder()
        .timeout_global(Some(DOWNLOAD_TIMEOUT))
        .tls_config(tls)
        .build();
    ureq::Agent::new_with_config(config)
}

/// 下载到 `dest`：先写 `<dest>.part`，校验非空后改名替换，失败不会留下半截文件
pub fn download(url: &str, dest: &Path) -> Result<(), String> {
    let agent = http_agent();
    let mut response = agent.get(url).call().map_err(|e| format!("请求 {url} 失败: {e}"))?;
    let status = response.status();
    if !status.is_success() {
        return Err(format!("{url} 返回状态码 {status}"));
    }

    let part = part_path(dest);
    let mut file = std::fs::File::create(&part).map_err(|e| format!("创建 {} 失败: {e}", part.display()))?;
    let mut reader = response.body_mut().as_reader();
    let mut buffer = vec![0u8; 64 * 1024];
    let mut written = 0u64;
    loop {
        let size = reader.read(&mut buffer).map_err(|e| format!("下载 {url} 失败: {e}"))?;
        if size == 0 {
            break;
        }
        file.write_all(&buffer[..size]).map_err(|e| format!("写入 {} 失败: {e}", part.display()))?;
        written += size as u64;
    }
    file.flush().map_err(|e| format!("写入 {} 失败: {e}", part.display()))?;
    drop(file);

    if written == 0 {
        let _ = std::fs::remove_file(&part);
        return Err(format!("{url} 返回空文件"));
    }
    replace_file(&part, dest)
}

/// 从 zip 里取出 `.onnx` 写到 `out_path`
pub fn extract_onnx(zip_path: &Path, out_path: &Path) -> Result<(), String> {
    let bytes = std::fs::read(zip_path).map_err(|e| format!("读取压缩包 {} 失败: {e}", zip_path.display()))?;
    let entry = find_onnx_entry(&bytes)?;
    let data = entry_data(&bytes, &entry)?;

    let part = part_path(out_path);
    {
        let mut file = std::fs::File::create(&part).map_err(|e| format!("创建 {} 失败: {e}", part.display()))?;
        file.write_all(&data).map_err(|e| format!("写入 {} 失败: {e}", part.display()))?;
        file.flush().map_err(|e| format!("写入 {} 失败: {e}", part.display()))?;
    }
    replace_file(&part, out_path)
}

/// zip 中央目录里的一条记录（只保留取数据需要的字段）
#[derive(Debug)]
struct ZipEntry {
    name: String,
    /// 0 = stored，8 = deflate
    method: u16,
    crc: u32,
    compressed_size: u32,
    uncompressed_size: u32,
    local_offset: u32,
}

impl ZipEntry {
    fn is_onnx(&self) -> bool {
        self.name.to_ascii_lowercase().ends_with(".onnx")
    }
}

fn part_path(dest: &Path) -> PathBuf {
    let mut part = dest.as_os_str().to_os_string();
    part.push(".part");
    PathBuf::from(part)
}

/// Windows 的改名不会覆盖已存在的目标，所以先删再改名
fn replace_file(part: &Path, dest: &Path) -> Result<(), String> {
    if let Some(parent) = dest.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent).map_err(|e| format!("创建目录 {} 失败: {e}", parent.display()))?;
        }
    }
    let _ = std::fs::remove_file(dest);
    std::fs::rename(part, dest).map_err(|e| {
        let _ = std::fs::remove_file(part);
        format!("保存 {} 失败: {e}", dest.display())
    })
}

/// 从文件尾部找到中央目录，返回第一个 `.onnx` 条目
fn find_onnx_entry(zip: &[u8]) -> Result<ZipEntry, String> {
    let search_from = zip.len().saturating_sub(MAX_EOCD_SEARCH);
    let eocd = (search_from..zip.len().saturating_sub(21))
        .rev()
        .find(|&at| at + 4 <= zip.len() && le32(zip, at) == EOCD_SIGNATURE)
        .ok_or("不是有效的 zip 文件（未找到中央目录结尾记录）")?;

    let mut cursor = le32(zip, eocd + 16) as usize;
    while cursor + 46 <= zip.len() && le32(zip, cursor) == CENTRAL_SIGNATURE {
        let name_len = le16(zip, cursor + 28) as usize;
        let extra_len = le16(zip, cursor + 30) as usize;
        let comment_len = le16(zip, cursor + 32) as usize;
        let name_start = cursor + 46;
        let name_end = name_start + name_len;
        if name_end > zip.len() {
            return Err("zip 中央目录损坏".to_string());
        }
        let entry = ZipEntry {
            name: String::from_utf8_lossy(&zip[name_start..name_end]).into_owned(),
            method: le16(zip, cursor + 10),
            crc: le32(zip, cursor + 16),
            compressed_size: le32(zip, cursor + 20),
            uncompressed_size: le32(zip, cursor + 24),
            local_offset: le32(zip, cursor + 42),
        };
        if entry.is_onnx() {
            return Ok(entry);
        }
        cursor = name_end + extra_len + comment_len;
    }
    Err("压缩包里没有 .onnx 文件".to_string())
}

/// 按中央目录记录的偏移读本地文件头，取出条目数据并按 CRC32 校验
fn entry_data(zip: &[u8], entry: &ZipEntry) -> Result<Vec<u8>, String> {
    if entry.compressed_size == ZIP64_MARKER || entry.uncompressed_size == ZIP64_MARKER {
        return Err(format!("暂不支持 ZIP64 压缩包（条目 {}）", entry.name));
    }
    let at = entry.local_offset as usize;
    if at + 30 > zip.len() || le32(zip, at) != LOCAL_SIGNATURE {
        return Err(format!("zip 本地文件头损坏（条目 {}）", entry.name));
    }
    let start = at + 30 + le16(zip, at + 26) as usize + le16(zip, at + 28) as usize;
    let end = start + entry.compressed_size as usize;
    if end > zip.len() {
        return Err(format!("zip 数据超出文件长度（条目 {}）", entry.name));
    }
    let raw = &zip[start..end];

    let data = match entry.method {
        0 => raw.to_vec(),
        8 => {
            let mut decoded = Vec::with_capacity(entry.uncompressed_size as usize);
            DeflateDecoder::new(raw)
                .read_to_end(&mut decoded)
                .map_err(|e| format!("解压 {} 失败: {e}", entry.name))?;
            decoded
        }
        other => return Err(format!("不支持的 zip 压缩方式: {other}"))
    };

    if data.len() != entry.uncompressed_size as usize {
        return Err(format!("{} 解压后大小不符: 期望 {}，实际 {}", entry.name, entry.uncompressed_size, data.len()));
    }
    let mut crc = flate2::Crc::new();
    crc.update(&data);
    if crc.sum() != entry.crc {
        return Err(format!("{} 的 CRC32 校验失败", entry.name));
    }
    Ok(data)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn endpoint_list_respects_overrides() {
        assert_eq!(
            endpoints(None, None),
            vec!["https://hf-mirror.com".to_string(), "https://huggingface.co".to_string()]
        );
        // CNOCR_MODEL_ENDPOINT 优先，并且只用它
        assert_eq!(endpoints(Some("https://mirror.example/"), Some("https://hf.example"))
            .as_slice(), ["https://mirror.example"]);
        // HF_ENDPOINT 排在默认镜像前面，重复项去掉
        assert_eq!(
            endpoints(None, Some("https://hf-mirror.com")).as_slice(),
            ["https://hf-mirror.com", "https://huggingface.co"]
        );
        assert_eq!(endpoints(Some(""), None).as_slice(), ["https://hf-mirror.com", "https://huggingface.co"]);
    }

    #[test]
    fn archive_url_is_composed_like_the_cpp_registry() {
        let url = archive_url(
            "https://hf-mirror.com/",
            "models/cnstd/1.2",
            "ch_PP-OCRv3_det_infer-onnx.zip"
        );
        assert_eq!(
            url,
            "https://hf-mirror.com/breezedeus/cnstd-cnocr-models/resolve/main/models/cnstd/1.2/ch_PP-OCRv3_det_infer-onnx.zip"
        );
        assert!(archive_url("https://a", "", "b.zip") == "https://a/breezedeus/cnstd-cnocr-models/resolve/main/b.zip");
    }

    fn le_bytes(out: &mut Vec<u8>, value: u16) {
        out.extend_from_slice(&value.to_le_bytes());
    }

    fn le4(out: &mut Vec<u8>, value: u32) {
        out.extend_from_slice(&value.to_le_bytes());
    }

    /// 手工拼一个 zip：`stored` 为 false 的条目用 raw deflate 压缩
    fn build_zip(entries: &[(&str, &[u8], bool)]) -> Vec<u8> {
        let mut files: Vec<u8> = Vec::new();
        let mut central: Vec<u8> = Vec::new();
        for (name, payload, stored) in entries {
            let method = if *stored { 0 } else { 8 };
            let data = if *stored {
                payload.to_vec()
            } else {
                let mut encoded = Vec::new();
                flate2::bufread::DeflateEncoder::new(*payload, flate2::Compression::default())
                    .read_to_end(&mut encoded)
                    .expect("构造测试用 deflate 数据失败");
                encoded
            };
            let mut crc = flate2::Crc::new();
            crc.update(payload);

            let local_offset = files.len() as u32;
            let mut record = Vec::new();
            le4(&mut record, CENTRAL_SIGNATURE);
            // 中央目录头：版本 x2、标志、method、时间+日期、crc、压缩后/原始大小、
            // 名字/额外字段/注释长度、磁盘号、内部/外部属性、本地头偏移
            le_bytes(&mut record, 20);
            le_bytes(&mut record, 20);
            le_bytes(&mut record, 0);
            le_bytes(&mut record, method);
            le_bytes(&mut record, 0);
            le_bytes(&mut record, 0);
            le4(&mut record, crc.sum());
            le4(&mut record, data.len() as u32);
            le4(&mut record, payload.len() as u32);
            le_bytes(&mut record, name.len() as u16);
            le_bytes(&mut record, 0);
            le_bytes(&mut record, 0);
            le_bytes(&mut record, 0);
            le_bytes(&mut record, 0);
            // 外部文件属性占 4 字节，之后才是本地头偏移
            le4(&mut record, 0);
            le4(&mut record, local_offset);
            record.extend_from_slice(name.as_bytes());
            central.extend_from_slice(&record);

            let mut local = Vec::new();
            le4(&mut local, LOCAL_SIGNATURE);
            le_bytes(&mut local, 20);
            le_bytes(&mut local, 0);
            le_bytes(&mut local, method);
            // 修改时间 + 修改日期
            le_bytes(&mut local, 0);
            le_bytes(&mut local, 0);
            le4(&mut local, crc.sum());
            le4(&mut local, data.len() as u32);
            le4(&mut local, payload.len() as u32);
            le_bytes(&mut local, name.len() as u16);
            le_bytes(&mut local, 0);
            local.extend_from_slice(name.as_bytes());
            local.extend_from_slice(&data);
            files.extend_from_slice(&local);
        }
        let directory_offset = files.len() as u32;
        files.extend_from_slice(&central);
        let mut eocd = Vec::new();
        le4(&mut eocd, EOCD_SIGNATURE);
        le_bytes(&mut eocd, 0);
        le_bytes(&mut eocd, 0);
        le_bytes(&mut eocd, entries.len() as u16);
        le_bytes(&mut eocd, entries.len() as u16);
        le4(&mut eocd, central.len() as u32);
        le4(&mut eocd, directory_offset);
        le_bytes(&mut eocd, 0);
        files.extend_from_slice(&eocd);
        files
    }

    #[test]
    fn finds_and_inflates_onnx_entry() {
        let payload: Vec<u8> = (0..300u16).map(|i| (i % 251) as u8).collect();
        let zip = build_zip(&[("readme.txt", b"ignore me", true), ("ppocr/model.onnx", &payload, false)]);
        let entry = find_onnx_entry(&zip).expect("应找到 .onnx 条目");
        assert_eq!(entry.name, "ppocr/model.onnx");
        assert_eq!(entry.method, 8);
        assert_eq!(entry_data(&zip, &entry).expect("解压失败"), payload);
    }

    #[test]
    fn handles_stored_entries_and_rejects_other_archives() {
        let payload = vec![7u8; 64];
        let zip = build_zip(&[("nested/a.onnx", &payload, true)]);
        let entry = find_onnx_entry(&zip).expect("应找到 stored 条目");
        assert_eq!(entry.method, 0);
        assert_eq!(entry_data(&zip, &entry).expect("读取失败"), payload);

        let no_onnx = build_zip(&[("notes.txt", b"only text", true)]);
        assert!(find_onnx_entry(&no_onnx).unwrap_err().contains(".onnx"));
        assert!(find_onnx_entry(b"not a zip at all").is_err());
    }

    #[test]
    fn detects_crc_mismatch() {
        let payload = vec![3u8; 40];
        let mut zip = build_zip(&[("bad.onnx", &payload, true)]);
        let entry = find_onnx_entry(&zip).expect("应解析出条目");
        let at = entry.local_offset as usize;
        let data_start = at + 30 + le16(&zip, at + 26) as usize + le16(&zip, at + 28) as usize;
        // 只篡改条目数据，中央目录里的 CRC 不动
        zip[data_start + 5] = 9;
        let entry = find_onnx_entry(&zip).expect("应解析出条目");
        assert!(entry_data(&zip, &entry).unwrap_err().contains("CRC32"));
    }

    /// 只检查配置构造成功且用的是 native-tls；真实 https 请求需要联网
    #[test]
    fn http_agent_uses_native_tls() {
        let agent = http_agent();
        assert_eq!(agent.config().tls_config().provider(), ureq::tls::TlsProvider::NativeTls);
    }

    /// 需要联网：真实下载检测模型并和仓库里的文件比对
    #[test]
    #[ignore = "需要联网，手动运行：cargo test --release -- --ignored"]
    fn downloads_real_model_archive() {
        let dir = std::env::temp_dir().join("cnocr_rs_download_test");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("创建临时目录失败");
        let zip = dir.join("det.zip");
        let onnx = dir.join("ch_PP-OCRv3_det_infer.onnx");
        let url = archive_url("https://hf-mirror.com", "models/cnstd/1.2", "ch_PP-OCRv3_det_infer-onnx.zip");
        download(&url, &zip).expect("下载失败");
        extract_onnx(&zip, &onnx).expect("解包失败");
        assert_eq!(std::fs::metadata(&onnx).expect("读取大小失败").len(), 2_429_472);
        // 仓库里已有同一份模型时直接逐字节比对
        let local = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("ch_PP-OCRv3_det_infer.onnx");
        if local.is_file() {
            let expected = std::fs::read(&local).expect("读取本地模型失败");
            assert_eq!(expected, std::fs::read(&onnx).expect("读取解包结果失败"));
        }
        std::fs::remove_dir_all(&dir).ok();
    }
}