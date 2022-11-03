
use std::{
    num::{NonZeroUsize},
    ops::{Index,IndexMut},
    marker::{PhantomData},
};

use crate::{
    Classification,
    inner::{
        data_scalar::{DataScalar},
        point::{Point},
    }
};

/// Tracks if we have/haven't visisted this cell yet
pub struct StateCell<'a,S: DataScalar,P: Point<S>> {
    visisted: bool,
    class: Classification,
    point: &'a P,
    marker: PhantomData<S>,
}
impl<'a,S: DataScalar,P:Point<S>> StateCell<'a,S,P> {
    pub(in crate) fn new(point: &'a P) -> Self {
        Self {
            visisted: false,
            class: Classification::Noise,
            point: point,
            marker: PhantomData,
        }
    }

    pub fn get_point(&self) -> &'a P {
        self.point
    }

    pub fn was_visisted(&self) -> bool {
        self.visisted
    }

    pub fn set_visisted(&mut self) {
        self.visisted = true;
    }

    pub fn is_noise(&self) -> bool {
        self.class == Classification::Noise
    }

    pub fn set_class(&mut self, class: Classification) {
        self.class = class;
    }
}

/// Handles the book keeping of if we visited
/// this node & what we need to know about this
/// that address.
#[derive(Default)]
pub struct StateTracker<'a, S: DataScalar, P: Point<S>> {
    state: Vec<StateCell<'a,S,P>>,
}
impl<'a, S: DataScalar, P: Point<S>> StateTracker<'a,S,P> {

    pub(in crate) fn new<I>(iter: I) -> Self
    where
        I: IntoIterator<Item=&'a P>,
    {
        Self {
            state: iter.into_iter().map(|x| StateCell::new(x)).collect()
        }
    }

    pub(in crate) fn range_iter(&self) -> impl Iterator<Item=NonZeroUsize> {
        (0usize..self.state.len()).filter_map(|x| NonZeroUsize::new(x+1))
    }
}
impl<'a, S: DataScalar, P: Point<S>> Index<NonZeroUsize> for StateTracker<'a,S,P> {
    type Output = StateCell<'a,S,P>;
    fn index<'b>(&'b self, idx: NonZeroUsize) -> &'b Self::Output {
        &self.state[idx.get()-1]
    }
}
impl<'a, S: DataScalar, P: Point<S>> IndexMut<NonZeroUsize> for StateTracker<'a,S,P> {
    fn index_mut<'b>(&'b mut self, idx: NonZeroUsize) -> &'b mut Self::Output {
        &mut self.state[idx.get()-1]
    }
}

