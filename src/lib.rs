#![allow(dead_code, unused_imports)]
#[repr(C)]
struct Connxret{
    output_lengths:*const std::ffi::c_longlong,
    logits:*mut std::ffi::c_void,
}
#[repr(C)]
struct OnnxModle{
    onnxmodle:*mut std::ffi::c_void,

}
extern "C"{
    /// # connxmodleinit  
    /// 初始化模型
    fn connxmodleInit()->OnnxModle; 
    fn connxmodleRun( modle:&OnnxModle ,input_lengths:std::ffi::c_longlong,x_length:std::ffi::c_longlong,x:*const u8)->Connxret;
    fn connxmodleRelease( modle:&OnnxModle );
}
pub mod ocr{
    use opencv::prelude::{MatTraitConstManual, Boxed};

    use crate::{ connxmodleRun, OnnxModle,connxmodleInit,connxmodleRelease};

    pub struct Ocr{
        ctc_data:Vec<char>,
        modle:OnnxModle,
    }
    impl core::ops::Drop for Ocr {
        fn drop(&mut self) {
            println!("删除cnocr cpp调用");
            unsafe{
                connxmodleRelease(&self.modle);
            }
        }
    }
    impl Ocr {
        pub fn new()->Ocr{
            let mut data=Vec::<char>::new();
            let mut f1=std::fs::File::open("label_cn.txt").unwrap();
            let mut str=String::new();
            use std::io::prelude::*;
            if let Ok(_)=f1.read_to_string(&mut str){
                for i in str.chars(){
                    if i!='\n'{
                        data.push(i);
                    }
                }
            }
            unsafe{
                let retm=connxmodleInit();
                return Ocr{
                    ctc_data:data,
                    modle:retm,
                }
            }
        }
    }
    fn softmax( matdata:opencv::core::Mat)->opencv::core::Mat{
        use opencv::prelude::MatTraitConst;
        use opencv::prelude::MatExprTraitConst;
        use opencv::core::prelude::*;
        //let mut sum=0.0;
        let mut ncdata=opencv::core::Mat::default();
        for i in 0..matdata.rows(){
            let mut tmp=matdata.row(i).unwrap();
            let mut out=opencv::core::Mat::default();
            opencv::core::reduce(&tmp, &mut out, 0, opencv::core::REDUCE_MAX, -1).unwrap();
            let maxvec:Vec<Vec<f32>>=out.to_vec_2d().unwrap();
            let max=maxvec[0][0] as f64;
            let one=opencv::core::Mat::ones(tmp.rows(), tmp.cols(), opencv::core::CV_32FC1).unwrap();
            if let opencv::core::MatExprResult::Ok(x)=tmp.clone()-max*one{
                opencv::core::exp(&x, &mut tmp).unwrap();
            }
            let sum=opencv::core::sum_elems(&tmp).unwrap();
            if let  opencv::core::MatExprResult::Ok(x)=tmp.clone()/sum[0] {
                ncdata.push_back(&x.to_mat().unwrap()).unwrap();
            }
        }
        ncdata
    }
    pub trait OcrTraitConst {
        fn from_mat(&self,imgdata:opencv::core::Mat)->Vec<(String,f32)>;
        fn from_path(&self,path:String)->Vec<(String,f32)>;
        fn ocr_for_single_lines(&self,inimgs:Vec<opencv::core::Mat>)->Vec<(String,f32)>;
        fn line_split(&self,inimg:opencv::core::Mat)->Vec<opencv::core::Mat>;
        fn ctc_best(&self,data:Vec<usize>)->String;
    }
    impl OcrTraitConst for Ocr{
        /**
        # 从opencv Mat识别文字
        > 输入： 因该是一个RGB彩色图片
        输出 识别出来的文字 和 准确率
        */
        fn from_mat(&self,imgdata:opencv::core::Mat)->Vec<(String,f32)>{
            use opencv::prelude::MatTraitConst;
            use opencv::core::prelude::*;
            let mut outimg=opencv::core::Mat::default();
            if let Err(_)=opencv::imgproc::cvt_color(&imgdata, &mut outimg,opencv::imgproc::COLOR_RGB2GRAY,0){
                assert!(false,"From_mat.cvt_color(imgdata,outimg) faild");
            }
            let mut res:Vec<(String,f32)>=Vec::new();
            let imgcol=outimg.cols();
            let imgrow=outimg.rows();
            if std::cmp::min(imgcol,imgrow)<2{
                return res;
            }
            if opencv::core::sum_elems(&outimg.col(0).unwrap()).unwrap()[0]<145.0{
                let o=opencv::core::Mat::ones(imgrow, imgcol, opencv::core::CV_8UC1).unwrap();
                if let opencv::core::MatExprResult::Ok(i)=(255f64*o)-&outimg{
                    outimg=i.to_mat().unwrap();
                }
            }
            let imgs=self.line_split(outimg);
            res=self.ocr_for_single_lines(imgs);
            res
        }
        /**
        # 从文件路径 读取图片 识别文字
        > 输入： 因该是 图片文件路径
        输出 识别出来的文字 和 准确率
        */
        fn from_path(&self,path:String)->Vec<(String,f32)>{
            use opencv::prelude::MatTraitConst;
            let inimg =opencv::imgcodecs::imread(path.as_str(), opencv::imgcodecs::IMREAD_COLOR).unwrap();
            return self.from_mat(inimg);
        }
        /**
        # 识别单行文字
        > 输入： 因该是一组 切分好行 的 opencv Mat 格式
        输出 识别出来的文字 和 准确率
        */
        fn  ocr_for_single_lines(&self,inimgs:Vec<opencv::core::Mat>)->Vec<(String,f32)>{
            let mut res:Vec<(String,f32)>=Vec::new();
            if inimgs.len()==0{
                return res;
            }
            use opencv::prelude::MatTraitConst;
            use opencv::core::prelude::*;
            for img in inimgs{
                let _imgcol=img.cols();
                let imgrow=img.rows();
                let mut imgmat=opencv::core::Mat::default();
                let ratio=32.0/imgrow as f64;
                let _sz=opencv::core::Size::new(ratio as i32, 32);
                opencv::imgproc::resize(&img,&mut imgmat,opencv::core::Size::default(), ratio , ratio,  0).unwrap();
                //println!("copyto modle imgsize({},{})",imgmat.cols(),imgmat.rows());//测试将进入模型的数据是否高度为32
                let mut  matdata;
                unsafe{
                    let sz=imgmat.size().unwrap();
                    let ret=connxmodleRun( &self.modle,sz.width as i64,(sz.height*sz.width) as i64, imgmat.data());
                    let length=*ret.output_lengths;
                    //println!("connx:{}",length);
                    matdata=opencv::core::Mat::new_rows_cols_with_data(length as i32, 6674, opencv::core::CV_32FC1, ret.logits, 0).unwrap();
                }
                
                //接下来用softmax处理信息
                matdata=softmax(matdata);
                
                //vargmax
                let mut resvec:Vec<usize>=Vec::new();
                let ncdata:Vec<Vec<f32>>=matdata.to_vec_2d().unwrap();
                for i in ncdata{
                    let mut maxvalue=0f32;
                    let mut maxvaluelocal=0usize;
                    for o in 0..i.len(){
                        if i[o]>maxvalue{
                            maxvalue=i[o];
                            maxvaluelocal=o;
                        }
                    }
                    resvec.push(maxvaluelocal);
                }
                //准确率
                let mut maxmat=opencv::core::Mat::default();
                let mut minmat=opencv::core::Mat::default();
                
                opencv::core::reduce(&matdata, &mut maxmat, 1, opencv::core::REDUCE_MAX, -1).unwrap();
                opencv::core::reduce(&maxmat, &mut minmat, 0, opencv::core::REDUCE_MIN, -1).unwrap();
                //println!("minmat.size:{},{}",minmat.cols(),minmat.rows());
                //匹配数据ctcbest
                let retstr=self.ctc_best(resvec);
                let min2d:&[f32]=minmat.data_typed().unwrap();
                println!("minmat size:{:?}",minmat.size().unwrap());
                res.push((retstr,min2d[0]));
            }
           
            res
        }
        /**
        # line_split
        > 将黑白图片每行找出最大值，列出竖直列，其中较大部分为文字。将其切分成 图片 每张图片仅只能有一行文字。
        > 所以 输入图片不能有 划线 类似表格等 如有倾斜也会导致无法划分图片切片
        输出 切分好的 图片 列表
        */
        fn line_split(&self,inimg:opencv::core::Mat)->Vec<opencv::core::Mat>{

            let mut list:Vec<opencv::core::Mat>=Vec::new();
            use opencv::prelude::MatTraitConst;
            use opencv::core::prelude::*;
            use opencv::prelude::*;
            use opencv::core::*;
            
            let imgcol=inimg.cols();
            let imgrow=inimg.rows();
            //println!("img({},{})",imgcol,imgrow);
            let mut tmp=opencv::core::Mat::default();

            opencv::core::reduce_arg_min(&inimg, &mut tmp, 1, false).unwrap();
            let tmpdata=tmp.data_bytes().unwrap();
            let mut splitlinedata:Vec<u8>=Vec::new();
            for i in 0..(tmpdata.len()/4){
                splitlinedata.push(tmpdata[i]+tmpdata[i+1]+tmpdata[i+2]+tmpdata[i+3]);
            }
            //println!("splitlinedata:{}:{:?}",splitlinedata.len(),splitlinedata);
            let mut lineforchar=0;
            let mut lineforcharstart=0;
            for i in 0..splitlinedata.len(){
                if splitlinedata[i]<100{
                    if lineforchar+1==i{
                        lineforchar+=1;
                    }
                    else{
                        if lineforchar-lineforcharstart>7{
                            let mut start=lineforcharstart;
                            let mut end=lineforchar;
                            if (start>0){
                                //start-=1;
                            }
                            let mut tmpvec:opencv::core::Vector<Range>=opencv::core::Vector::new();
                            tmpvec.push(Range::new(start as i32, end as i32).unwrap());
                            tmpvec.push(Range::all().unwrap());
                            let mut tmpimg=opencv::core::Mat::default();
                            tmpimg=opencv::core::Mat::ranges( &inimg, &tmpvec).unwrap();
                            list.push(tmpimg);
                        
                        }
                        lineforcharstart=i;
                        lineforchar=i;
                    }
                }
            }
            if lineforchar-lineforcharstart>5{
                let mut start=lineforcharstart;
                let mut end= lineforchar;
                if (start>3){
                    start-=3;
                }
                if end<(imgcol-1)as usize{
                    end+=1;
                }
                let mut tmpvec:opencv::core::Vector<Range>=opencv::core::Vector::new();
                tmpvec.push(Range::new(start as i32, end as i32).unwrap());
                tmpvec.push(Range::all().unwrap());
                let mut tmpimg=opencv::core::Mat::default();
                tmpimg=opencv::core::Mat::ranges(&inimg, &tmpvec).unwrap();
                list.push(tmpimg);
            }
            //for i in &list{
            //    println!("({},{})",i.cols(),i.rows());
            //    opencv::highgui::imshow("", i);
            //    opencv::highgui::wait_key(0);
            //}
            list
        }
        /**
        #合成 文字
        > 输入：文字列表
        > 功能：查询ai模型字典，将列表转换为字符串
        输出 处理好的 字符串
        */
        fn ctc_best(&self,data:Vec<usize>)->String{
            let mut res =String::new();
            let mut vui:Vec<usize>=Vec::new();
            for i in data{
                if vui.len()!=0{
                    if vui[vui.len()-1]!=i{
                        vui.push(i);
                    }
                }
                else {
                    vui.push(i);
                }
            }
            for i in vui{
                if i<6642{
                    res.push(self.ctc_data[i]);
                }
            }
            res
        }
    }
}