use colored::Colorize;

pub(crate) fn rgb_from_val(val: f32) -> (u8, u8, u8){
    let v = val.clamp(-1.0, 1.0);
    let r = 255 + std::cmp::min(0, (255.0 * v).ceil() as i32);
    let g = (255.0 - 255.0 * v.abs()).ceil() as u8;
    let b = 255 + std::cmp::min(0, (255.0 * -v).ceil() as i32);
    (r as u8, g, b as u8)
}

pub(crate) fn print_square_from_val(val: f32){
    let (r, g, b) = rgb_from_val(val);
    let s = String::from("■").truecolor(r, g, b);
    print!("{}", s);
}

pub(crate) fn print_colored_matrix<const F: usize, const D: usize>(mat: &[[f32; F]; D]){
    for arr in mat.iter(){
        for val in arr.iter(){
            print_square_from_val(*val);
            print!(" ");
        }
        println!();
    }
}

pub(crate) fn print_colored_vector<const F: usize>(arr: &[f32; F]){
    for val in arr.iter(){
        print_square_from_val(*val);
        print!(" ");
    }
    println!();
    println!();
}

pub(crate) fn pprint<const F: usize, const D: usize>
    (arr: &[[f32; F]; D]){
    for row in arr.iter() {
        println!("{:.2?}", row)
    }
}
