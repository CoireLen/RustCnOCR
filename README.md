# RustCnOcr

- 一份来自python cnocr的Rust实现，官方文档->[CnOcr文档](https://cnocr.readthedocs.io/zh/latest/)
- 全部依赖都来自 crates.io，不需要再安装任何本地 C/C++ 库
- 识别流程参考了同作者的 C++ 实现 CnocrCpp：多模型注册表、按 token 读取字典、
  依据模型元数据自动组装输入输出、投影切分文字行

## 依赖

- `ort`：ONNX Runtime 的 Rust 绑定，构建时自动下载对应平台的 onnxruntime 预编译库（`download-binaries`），
  不再需要手动下载 C++ 版 onnxruntime、也不需要改 build.rs
  - 版本暂用 `=2.0.0-rc.13` 固定，等 ort 2.0.0 正式版发布后可以放宽；rc.15 起要求 rustc 1.92
  - 计算后端为 CPU；原 C++ 实现里的 TensorRT 加速没有迁移，需要 GPU 时给 `ort` 打开 `cuda` / `tensorrt` 特性
  - 内网/离线环境可以设 `ORT_LIB_LOCATION` 指向已有的 onnxruntime，或改用 `load-dynamic` 特性
- `image`：图片解码与缩放，纯 Rust 实现，替换了原来的 `opencv` 依赖

## 模型与字典

模型文件（`.onnx`）按以下顺序查找，取第一个存在的：`modle/` → 当前目录 → `<CNOCR_HOME>/models`
（未设 `CNOCR_HOME` 时，Windows 用 `%APPDATA%\cnocr`，其他平台用 `~/.cnocr`）。
字典文件同样依次查找 `dict/` → 当前目录。

内置可选的识别模型（模型文件需自行放置，可从
[hf-mirror 的 breezedeus/cnstd-cnocr-models](https://hf-mirror.com/breezedeus/cnstd-cnocr-models) 下载）：

| 模型名 | 通道 | 输入高度 | 字典 | 说明 |
| --- | --- | --- | --- | --- |
| `cnocr136fc` | 1 | 32 | `label_cn.txt` | 早期中文识别模型（cnocr 2.2），默认 |
| `densenet_lite_136-gru` | 1 | 32 | `label_cn_2.3.txt` | 通用中文识别模型（cnocr 2.3） |
| `scene-densenet_lite_136-gru` | 1 | 32 | `label_cn_2.3.txt` | 场景文字识别模型 |
| `doc-densenet_lite_136-gru` | 1 | 32 | `label_cn_2.3.txt` | 文档识别模型 |
| `number-densenet_lite_136-fc` | 1 | 32 | `label_number.txt` | 数字识别模型 |
| `en_number_mobile_v2.0` | 3 | 32 | `en_dict.txt` | 英文数字识别模型（PaddleOCR） |
| `chinese_cht_PP-OCRv3` | 3 | 32 | `chinese_cht_dict.txt` | 繁体中文识别模型 |
| `ch_PP-OCRv3` | 3 | 48 | `ppocr_keys_v1.txt` | 简体中英文识别模型 |
| `ch_ppocr_mobile_v2.0` | 3 | 32 | `ppocr_keys_v1.txt` | 简体中英文识别模型（mobile v2.0） |

```rust
use cnocr_rs::ocr::{Ocr, OcrTraitConst};
let ocr = Ocr::with_model_name("ch_PP-OCRv3");   // 也可用 Ocr::with_model(UseModel::ChPpOcrV3)
```

> 仓库自带 `cnocr136fc.onnx` 与 `label_cn.txt`，所以开箱即用；换模型时需要另外放置对应的模型和字典。
> 投影切分只处理白底黑字，深色背景会先自动反色。

## 与旧版本的接口差异

去掉 `opencv` 后，接口类型不再是 `Mat`：

| 旧接口 | 新接口 |
| --- | --- |
| `from_mat(Mat)` | `from_image(&DynamicImage)` |
| `from_path(String)` | `from_path(&str)` |
| `ocr_for_single_lines(Vec<Mat>)` | `ocr_for_single_lines(&[GrayImage])` |
| `line_split(Mat) -> Vec<Mat>` | `line_split(&GrayImage) -> Vec<GrayImage>` |
| `ctc_best(Vec<usize>)` | `ctc_best(&[usize])` |
| `Ocr::new()` | 仍是 `Ocr::new()`，另外新增 `with_model` / `with_model_name` / `available_models` |

`label_cn.txt` 换成了与 Python cnocr / CnocrCpp 一致的版本：占位标签写作 `<unused1>`…，
下标与字符严格对齐，`<space>` 会被读成空格。旧版把占位行存成了空格，识别结果里会夹进多余空格。

## 待解决问题

- PP-OCRv3 文本行检测（CnocrCpp 的 `cnstd`）没有迁移，倾斜、表格、场景图仍依赖投影切分
- 模型自动下载没有实现，需要手动放置模型文件

## 使用演示

```rust
use cnocr_rs::ocr::{Ocr, OcrTraitConst};
fn main() {
    let ocr = Ocr::new();
    let vs = ocr.from_path("cnocr.png");
    for i in vs {
        println!("识别到:{:?}", i);
    }
}
```
```sh
识别到:("cnocr自V2.1.2之后，可直接使用的模型包含两类：1）cnocr自己训练的模型，通常会包含PyTorch和", 0.5027224)
识别到:("ONNX版本；2）从其他ocr引擎搬运过来的训练好的外部模型，ONNX化后用于 cnocr中。", 0.71831757)
```

命令行示例支持指定图片和模型：`cargo run --release -- cnocr.png cnocr136fc`。
回归测试：`cargo test`（需要仓库根目录下的模型与字典）。
