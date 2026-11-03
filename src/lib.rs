//! 纯 Rust 的中文 OCR，推理走 `ort`（crates.io 上的 ONNX Runtime 绑定），图像处理只用 `image`。
//!
//! 用法见 `src/main.rs`：`Ocr::new()` 使用默认模型，`Ocr::with_model_name("ch_PP-OCRv3")`
//! 切换模型，模型与字典文件需要按 `model_registry` 说明放到 `modle/`、`dict/` 或当前目录。

pub mod model_registry;
pub mod ocr;
