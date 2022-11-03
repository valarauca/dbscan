
use std::{
    num::{NonZeroUsize},
    marker::{PhantomData},
};

use crate::inner::{
    point::Point,
    data_scalar::DataScalar,
};

/// Collection of point data (normally: [S;usize])
/// and an index to reference visited & classification information
pub struct RTreeItem<'a, S: DataScalar, P: Point<S>> {
    idx: NonZeroUsize,
    point: &'a P,
    marker: PhantomData<S>,
}
impl<'a, S: DataScalar, P: Point<S>> RTreeItem<'a, S,P> {
    pub(in crate) fn new(idx: NonZeroUsize, point: &'a P) -> Self {
        Self { idx, point, marker: PhantomData }
    }

    pub fn get_idx(&self) -> NonZeroUsize {
        self.idx.clone()
    }

    pub(in crate) fn get_point(&self) -> P {
        self.point.clone()
    }
}
impl<'a, S: DataScalar, P: Point<S>> Clone for RTreeItem<'a, S,P>
where
    S: Clone,
    P: Clone,
{
    fn clone(&self) -> Self {
        Self {
            idx: self.idx.clone(),
            point: self.point,
            marker: PhantomData,
        }
    }
}
impl<'a, S: DataScalar, P: Point<S>> rstar::RTreeObject for RTreeItem<'a,S,P>
{
    type Envelope = rstar::AABB<P>;
    fn envelope(&self) -> Self::Envelope {
        rstar::AABB::from_point(self.get_point())
    }
}
impl<'a, S: DataScalar, P: Point<S>> rstar::PointDistance for RTreeItem<'a,S,P>
{
    fn distance_2(&self, point: &P) -> S {
        self.point.distance(point)
    }

}
