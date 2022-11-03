
use std::slice::Iter;

use crate::inner::data_scalar::{DataScalar};

/// Defines a point in N-Dimensional space
pub trait Point<S: DataScalar>: rstar::Point<Scalar=S> {

    /// for iterating over the scalars within a point
    fn iter<'a>(&'a self) -> Iter<'a,S>;

    /// This returns distance squared, it doesn't do
    /// the square root step to avoid unnecessary cpu
    /// time
    #[inline(always)]
    fn distance<P>(&self, other: &P) -> S
    where
        P: Point<S> + ?Sized,
    {
        S::normalize(
            self.iter()
                .zip(other.iter())
                .fold(0f64, |acc, (&x, &y)| {
                    acc + (x.to_calc() - y.to_calc()).powi(2)
                })
        )
    }
}

/*
 * Implementations
 *
 */
impl<S: DataScalar> Point<S> for [S;2] {
    fn iter<'a>(&'a self) -> Iter<'a,S> {
        self.iter()
    }
}
impl<S: DataScalar> Point<S> for [S;3] {
    fn iter<'a>(&'a self) -> Iter<'a,S> {
        self.iter()
    }
}

impl<S: DataScalar> Point<S> for [S;4] {
    fn iter<'a>(&'a self) -> Iter<'a,S> {
        self.iter()
    }
}
impl<S: DataScalar> Point<S> for [S;5] {
    fn iter<'a>(&'a self) -> Iter<'a,S> {
        self.iter()
    }
}
impl<S: DataScalar> Point<S> for [S;6] {
    fn iter<'a>(&'a self) -> Iter<'a,S> {
        self.iter()
    }
}
impl<S: DataScalar> Point<S> for [S;7] {
    fn iter<'a>(&'a self) -> Iter<'a,S> {
        self.iter()
    }
}
impl<S: DataScalar> Point<S> for [S;8] {
    fn iter<'a>(&'a self) -> Iter<'a,S> {
        self.iter()
    }
}
impl<S: DataScalar> Point<S> for [S;9] {
    fn iter<'a>(&'a self) -> Iter<'a,S> {
        self.iter()
    }
}
