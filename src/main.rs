use RustCnOcr::Ocr::{ocr,OcrTraitConst};
fn main() {
    let ocr=ocr::new();
    let vs=ocr.from_path("/home/len/tmp/cnocr.png".to_string());
}
