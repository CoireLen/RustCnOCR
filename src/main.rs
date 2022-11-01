use RustCnOcr::Ocr::{ocr,OcrTraitConst};
fn main() {
    let ocr=ocr::new();
    let vs=ocr.from_path("cnocr.png".to_string());
    for i in vs{
        println!("识别到:{}",i);
    }
}
