//! 文本框几何计算，替代 C++ 实现里由 opencv 提供的部分
//! （`cv::minAreaRect`、`cv::contourArea`、`cv::arcLength`、`cv::fillPoly` 等）。
//! 约定：文本框四点顺序为左上、右上、右下、左下。

/// 平面上的点（概率图坐标或原图坐标，均为浮点像素坐标）
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Pt {
    pub x: f32,
    pub y: f32,
}

impl Pt {
    pub fn new(x: f32, y: f32) -> Pt {
        Pt { x, y }
    }
}

/// 检测到的文本框
#[derive(Clone, Copy, Debug)]
pub struct DetBox {
    /// 左上、右上、右下、左下
    pub points: [Pt; 4],
    /// 框内平均概率
    pub score: f32,
}

pub fn point_distance(a: Pt, b: Pt) -> f32 {
    (a.x - b.x).hypot(a.y - b.y)
}

/// 多边形面积绝对值（鞋带公式），对应 `std::abs(cv::contourArea)`
pub fn polygon_area(points: &[Pt]) -> f64 {
    let mut sum = 0.0f64;
    for i in 0..points.len() {
        let p = points[i];
        let q = points[(i + 1) % points.len()];
        sum += f64::from(p.x) * f64::from(q.y) - f64::from(q.x) * f64::from(p.y);
    }
    (sum * 0.5).abs()
}

/// 多边形周长，对应 `cv::arcLength(points, true)`
pub fn polygon_perimeter(points: &[Pt]) -> f64 {
    if points.len() < 2 {
        return 0.0;
    }
    (0..points.len())
        .map(|i| f64::from(point_distance(points[i], points[(i + 1) % points.len()])))
        .sum()
}

/// 单调链凸包，输入无需有序；返回值按逆时针排列且不含共线点
pub fn convex_hull(points: &[Pt]) -> Vec<Pt> {
    let mut sorted: Vec<Pt> = points.to_vec();
    sorted.sort_by(|a, b| {
        a.x.partial_cmp(&b.x)
            .unwrap_or(core::cmp::Ordering::Equal)
            .then(a.y.partial_cmp(&b.y).unwrap_or(core::cmp::Ordering::Equal))
    });
    sorted.dedup_by(|a, b| a.x == b.x && a.y == b.y);
    if sorted.len() < 3 {
        return sorted;
    }

    // 叉积：>0 左转，==0 共线
    let cross = |o: Pt, a: Pt, b: Pt| {
        f64::from(a.x - o.x) * f64::from(b.y - o.y) - f64::from(a.y - o.y) * f64::from(b.x - o.x)
    };

    let mut hull: Vec<Pt> = Vec::with_capacity(sorted.len() * 2);
    for p in &sorted {
        while hull.len() >= 2 && cross(hull[hull.len() - 2], *hull.last().unwrap(), *p) <= 0.0 {
            hull.pop();
        }
        hull.push(*p);
    }
    let lower_len = hull.len() + 1;
    for p in sorted[..sorted.len() - 1].iter().rev() {
        while hull.len() >= lower_len && cross(hull[hull.len() - 2], *hull.last().unwrap(), *p) <= 0.0 {
            hull.pop();
        }
        hull.push(*p);
    }
    hull.pop();
    hull
}

/// 最小面积外接矩形
#[derive(Clone, Copy, Debug)]
pub struct MinRect {
    /// 四个角点，沿矩形边依次排列
    pub corners: [Pt; 4],
    /// 短边长度
    pub side: f32,
}

/// 点集的最小外接矩形（`cv::minAreaRect`）。凸包退化成线段或点时返回 `None`，
/// 与 opencv 给出的零高度矩形一样会被后续的边长阈值过滤掉。
pub fn min_area_rect(points: &[Pt]) -> Option<MinRect> {
    let hull = convex_hull(points);
    if hull.len() < 3 {
        return None;
    }

    let mut best: Option<(f64, MinRect)> = None;
    for i in 0..hull.len() {
        let a = hull[i];
        let b = hull[(i + 1) % hull.len()];
        let (dx, dy) = (f64::from(b.x - a.x), f64::from(b.y - a.y));
        let len = dx.hypot(dy);
        if len < 1e-6 {
            continue;
        }
        // 以该条边为一个坐标轴，另一个轴取其法向，二者构成正交基
        let (ux, uy) = (dx / len, dy / len);
        let (vx, vy) = (-uy, ux);

        let (mut min_u, mut max_u, mut min_v, mut max_v) = (f64::MAX, f64::MIN, f64::MAX, f64::MIN);
        for p in &hull {
            let (px, py) = (f64::from(p.x), f64::from(p.y));
            let (pu, pv) = (px * ux + py * uy, px * vx + py * vy);
            min_u = min_u.min(pu);
            max_u = max_u.max(pu);
            min_v = min_v.min(pv);
            max_v = max_v.max(pv);
        }

        let width = max_u - min_u;
        let height = max_v - min_v;
        let area = width * height;
        if width < 1e-6 || height < 1e-6 {
            continue;
        }
        let better = match best {
            Some((best_area, _)) => area < best_area,
            None => true
        };
        if better {
            // 正交基下 P = (P·u)·u + (P·v)·v，直接把投影值还原成坐标
            let corner = |u: f64, v: f64| Pt::new((u * ux + v * vx) as f32, (u * uy + v * vy) as f32);
            let rect = MinRect {
                corners: [
                    corner(min_u, min_v),
                    corner(max_u, min_v),
                    corner(max_u, max_v),
                    corner(min_u, max_v)
                ],
                side: width.min(height) as f32
            };
            best = Some((area, rect));
        }
    }
    best.map(|(_, rect)| rect)
}

/// 与 PaddleOCR / `cv2.minAreaRect` 后处理一致：按 x 排序后还原左上、右上、右下、左下，
/// 同时返回短边长度
pub fn mini_boxes(points: &[Pt]) -> Option<([Pt; 4], f32)> {
    let rect = min_area_rect(points)?;
    let mut raw = rect.corners;
    raw.sort_by(|a, b| a.x.partial_cmp(&b.x).unwrap_or(core::cmp::Ordering::Equal));

    let (mut i1, mut i4) = (0, 1);
    if raw[1].y <= raw[0].y {
        i1 = 1;
        i4 = 0;
    }
    let (mut i2, mut i3) = (2, 3);
    if raw[3].y <= raw[2].y {
        i2 = 3;
        i3 = 2;
    }

    Some(([raw[i1], raw[i2], raw[i3], raw[i4]], rect.side))
}

/// 多边形等距外扩（miter 拐角），等价于 pyclipper 的 JT_ROUND 近似
pub fn offset_polygon(poly: &[Pt], distance: f32) -> Vec<Pt> {
    let n = poly.len();
    if n < 3 || distance <= 0.0 {
        return poly.to_vec();
    }

    let mut signed_area = 0.0f64;
    for i in 0..n {
        let p = poly[i];
        let q = poly[(i + 1) % n];
        signed_area += f64::from(p.x) * f64::from(q.y) - f64::from(q.x) * f64::from(p.y);
    }
    let orient = if signed_area >= 0.0 { 1.0f32 } else { -1.0f32 };

    // 边的外向法线
    let outward = |a: Pt, b: Pt| -> Pt {
        let (edge_x, edge_y) = (b.x - a.x, b.y - a.y);
        let length = edge_x.hypot(edge_y);
        if length < 1e-6 {
            return Pt::new(0.0, 0.0);
        }
        Pt::new(orient * edge_y / length, -orient * edge_x / length)
    };

    let mut out = vec![Pt::default(); n];
    for i in 0..n {
        let prev = poly[(i + n - 1) % n];
        let cur = poly[i];
        let next = poly[(i + 1) % n];
        let normal_prev = outward(prev, cur);
        let normal_next = outward(cur, next);
        let line1_start = Pt::new(prev.x + normal_prev.x * distance, prev.y + normal_prev.y * distance);
        let line2_start = Pt::new(cur.x + normal_next.x * distance, cur.y + normal_next.y * distance);
        let dir1 = Pt::new(cur.x - prev.x, cur.y - prev.y);
        let dir2 = Pt::new(next.x - cur.x, next.y - cur.y);

        let denom = f64::from(dir1.x) * f64::from(dir2.y) - f64::from(dir1.y) * f64::from(dir2.x);
        if denom.abs() < 1e-6 {
            out[i] = Pt::new(
                cur.x + (normal_prev.x + normal_next.x) * distance * 0.5,
                cur.y + (normal_prev.y + normal_next.y) * distance * 0.5
            );
            continue;
        }
        let diff = Pt::new(line2_start.x - line1_start.x, line2_start.y - line1_start.y);
        let t = (f64::from(diff.x) * f64::from(dir2.y) - f64::from(diff.y) * f64::from(dir2.x)) / denom;
        out[i] = Pt::new(line1_start.x + dir1.x * t as f32, line1_start.y + dir1.y * t as f32);
    }
    out
}

/// 把四点框栅格化成掩码（1 表示框内），对应 `cv::fillPoly`；
/// `points` 已经是相对 `(origin_x, origin_y)` 的整数坐标，掩码尺寸 `w*h`
pub fn fill_polygon(points: &[Pt; 4], mask: &mut [u8], w: usize, h: usize) {
    let inside = |px: f32, py: f32| -> bool {
        // 偶奇规则射线相交，凸四边形与 opencv 的非零填充结果一致
        let mut count = 0usize;
        for i in 0..4 {
            let a = points[i];
            let b = points[(i + 1) % 4];
            if (a.y > py) != (b.y > py) {
                let x_at = a.x + (py - a.y) * (b.x - a.x) / (b.y - a.y);
                if x_at > px {
                    count += 1;
                }
            }
        }
        count % 2 == 1
    };

    for y in 0..h {
        let py = y as f32 + 0.5;
        for x in 0..w {
            if inside(x as f32 + 0.5, py) {
                mask[y * w + x] = 1;
            }
        }
    }
}

/// 四点按左上、右上、右下、左下排序，对应 C++ 的 `order_points_clockwise`
pub fn order_points_clockwise(points: &[Pt; 4]) -> [Pt; 4] {
    let mut idx = [0usize, 1, 2, 3];
    idx.sort_by(|a, b| points[*a].x.partial_cmp(&points[*b].x).unwrap_or(core::cmp::Ordering::Equal));

    let mut left = [idx[0], idx[1]];
    let mut right = [idx[2], idx[3]];
    if points[left[1]].y < points[left[0]].y {
        left.swap(0, 1);
    }
    if points[right[1]].y < points[right[0]].y {
        right.swap(0, 1);
    }
    [points[left[0]], points[right[0]], points[right[1]], points[left[1]]]
}

/// 同一行的文本框按左右排序，不同行按上下排序
fn compare_boxes(a: &DetBox, b: &DetBox) -> bool {
    let (a_left, a_top, a_bottom) = (a.points[0].x, a.points[0].y, a.points[2].y);
    let (b_left, b_top, b_bottom) = (b.points[0].x, b.points[0].y, b.points[2].y);

    let mut y_overlap_ratio = 0.0f32;
    if !(a_bottom <= b_top || b_bottom <= a_top) {
        let overlap = a_bottom.min(b_bottom) - a_top.max(b_top);
        y_overlap_ratio = overlap / (a_bottom - a_top).min(b_bottom - b_top).max(1.0);
    }
    if y_overlap_ratio > 0.5 { a_left < b_left } else { a_top < b_top }
}

/// `compare_boxes` 不是严格弱序（是否比较左右取决于两框的纵向重叠），不能用 `sort_unstable`；
/// 这里用插入排序保证结果确定，与 Python 的实现思路一致
pub fn sort_reading_order(boxes: &mut [DetBox]) {
    for i in 1..boxes.len() {
        let key = boxes[i];
        let mut j = i;
        while j > 0 && compare_boxes(&key, &boxes[j - 1]) {
            boxes[j] = boxes[j - 1];
            j -= 1;
        }
        boxes[j] = key;
    }
}
