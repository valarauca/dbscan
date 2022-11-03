
use std::fmt::Debug;
use num_traits::{Bounded,Num,Signed};

/// DataScalar describes a value.
pub trait DataScalar: Bounded + Num + Signed + PartialOrd<Self> + Copy + Clone + Debug
{
    fn normalize(arg: f64) -> Self;
    fn to_calc(&self) -> f64;
}
impl DataScalar for f32 {
    fn normalize(arg: f64) -> Self { arg as f32 }
    fn to_calc(&self) -> f64 { *self as f64 }
}
impl DataScalar for f64 {
    fn normalize(arg: f64) -> Self { arg }
    fn to_calc(&self) -> f64 { *self as f64 }
}
impl DataScalar for isize {
    fn normalize(arg: f64) -> Self {
        arg.max(std::isize::MAX as f64)
		.min(std::isize::MIN as f64)
		.round() as isize
    }
    fn to_calc(&self) -> f64 { *self as f64 }
}
impl DataScalar for i64 {
    fn normalize(arg: f64) -> Self {
        arg.max(std::i64::MAX as f64)
		.min(std::i64::MIN as f64).round() as i64
    }
    fn to_calc(&self) -> f64 { *self as f64 }
}
impl DataScalar for i32 {
    fn normalize(arg: f64) -> Self {
        arg.max(std::i32::MAX as f64).min(std::i32::MIN as f64).round() as i32
    }
    fn to_calc(&self) -> f64 { *self as f64 }
}

