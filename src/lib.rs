//! # A Density-Based Algorithm for Discovering Clusters
//!
//! This algorithm finds all points within `eps` distance of each other and
//! attempts to cluster them. If there are at least `mpt` points reachable
//! (within distance `eps`) from a given point P, then all reachable points are
//! clustered together. The algorithm then attempts to expand the cluster,
//! finding all border points reachable from each point in the cluster
//!
//!
//! See `Ester, Martin, et al. "A density-based algorithm for discovering
//! clusters in large spatial databases with noise." Kdd. Vol. 96. No. 34.
//! 1996.` for the original paper
//!
//! Thanks to the rusty_machine implementation for inspiration

use std::fmt::Debug;

extern crate num_traits;
use num_traits::{Bounded,Num,Signed};

use Classification::{Core, Edge, Noise};

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

/// Point is a collection of Dimensions (`T`)
pub trait Point<T: DataScalar>:  rstar::Point<Scalar=T>
where
    for<'a> &'a Self: IntoIterator<Item=&'a T>,
{
    #[inline(always)]
    fn distance<A,B>(&self, other: &A) -> T
    where
        B: DataScalar,
        for<'b> &'b A: IntoIterator<Item=&'b B>,
        A: Point<B> + ?Sized,
    {

        // no sqrt of distance is taken per rstar trees requirement
        T::normalize(
            self.into_iter()
                .zip(other.into_iter())
                .fold(0f64, |acc, (&x, &y)| {
                    acc + (x.to_calc() - y.to_calc()).powi(2)
                })
        )
    }
}
impl<T: DataScalar> Point<T> for [T;2]
{ }
impl<T: DataScalar> Point<T> for [T;3]
{ }
impl<T: DataScalar> Point<T> for [T;4]
{ }
impl<T: DataScalar> Point<T> for [T;5]
{ }
impl<T: DataScalar> Point<T> for [T;6]
{ }
impl<T: DataScalar> Point<T> for [T;7]
{ }
impl<T: DataScalar> Point<T> for [T;8]
{ }
impl<T: DataScalar> Point<T> for [T;9]
{ }

/// Describes a collection of points. Traits for `Vec<T>` and `P<T>` are already implemented.
pub trait Population<P,T>:
where
    T: DataScalar,
    for<'a> &'a P: IntoIterator<Item=&'a T>,
    P: Point<T>,
    for<'b> &'b Self: IntoIterator<Item=&'b P>,
    Self: std::convert::AsRef<[P]> + std::ops::Index<usize,Output=P>,
{ 
    fn length(&self) -> usize {
        <Self as std::convert::AsRef<[P]>>::as_ref(self).len()
    }
}
impl<P,T> Population<P,T> for Vec<P>
where
    T: DataScalar,
    for<'a> &'a P: IntoIterator<Item=&'a T>,
    P: Point<T>,
{ }
impl<P,T> Population<P,T> for [P]
where
    T: DataScalar,
    for<'a> &'a P: IntoIterator<Item=&'a T>,
    P: Point<T>,
{ }

/// Classification according to the DBSCAN algorithm
#[derive(Debug, Copy, Clone, PartialEq, PartialOrd)]
pub enum Classification {
    /// A point with at least `min_points` neighbors within `eps` diameter
    Core(usize),
    /// A point within `eps` of a core point, but has less than `min_points` neighbors
    Edge(usize),
    /// A point with no connections
    Noise,
}

/// Cluster datapoints using the DBSCAN algorithm
///
/// # Arguments
/// * `eps` - maximum distance between datapoints within a cluster
/// * `min_points` - minimum number of datapoints to make a cluster
/// * `input` - a Vec<Vec<f64>> of datapoints, organized by row
pub fn cluster<T,P,C>(eps: T, min_points: usize, input: Vec<P>) -> Vec<Classification>
where
    T: DataScalar,
    for<'a> &'a P: IntoIterator<Item=&'a T>,
    P: Point<T>,
    for<'b> &'b C: IntoIterator<Item=&'b P>,
{
    let mut m = Model::new(eps, min_points,input);
    m.run()
}

struct RTreeItem<T: DataScalar, P: Point<T>>
where
    for<'a> &'a P: IntoIterator<Item=&'a T>,
{
    idx: usize,
    point: P,
}
impl<T: DataScalar, P: Point<T>> Clone for RTreeItem<T,P>
where
    T: Clone,
    P: Clone,
    for<'a> &'a P: IntoIterator<Item=&'a T>,
{
    fn clone(&self) -> Self {
        Self { idx: self.get_idx(), point: self.point.clone() }
    }
}
impl<T: DataScalar, P: Point<T>> rstar::PointDistance for RTreeItem<T,P>
where
    for<'a> &'a P: IntoIterator<Item=&'a T>,
{
    fn distance_2(&self, point: &P) -> T {
        self.point.distance(point)
    }
}
impl<T: DataScalar, P: Point<T>> RTreeItem<T,P>
where
    for<'a> &'a P: IntoIterator<Item=&'a T>,
{
    fn new(point: P, idx: usize) -> Self {
        Self { point, idx }
    }
    fn get_idx(&self) -> usize { self.idx.clone() }
}
impl<T: DataScalar, P: Point<T>> rstar::RTreeObject for RTreeItem<T,P>
where
    for<'a> &'a P: IntoIterator<Item=&'a T>,
{
    type Envelope = rstar::AABB<P>;
    fn envelope(&self) -> Self::Envelope {
        rstar::AABB::from_point(self.point.clone())
    }
}

/// DBSCAN parameters
pub struct Model<T: DataScalar, P: Point<T>>
where
    for<'a> &'a P: IntoIterator<Item=&'a T>,
{
    /// Epsilon value - maximum distance between points in a cluster
    pub eps: T,
    /// Minimum number of points in a cluster
    pub mpt: usize,
    tree: rstar::RTree<RTreeItem<T,P>>,
    classification: Vec<Classification>,
    population: Vec<P>,
    visited: Vec<bool>,
}

impl<T: DataScalar, P: Point<T>> Model<T,P>
where
    for<'a> &'a P: IntoIterator<Item=&'a T>,
{
    /// Create a new `Model` with a set of parameters
    ///
    /// # Arguments
    /// * `eps` - maximum distance between datapoints within a cluster
    /// * `min_points` - minimum number of datapoints to make a cluster
    pub fn new(eps: T, min_points: usize, population: Vec<P>) -> Model<T,P>
    {
        let classification = (0..population.len()).map(|_| Noise).collect();
        let visited = (0..population.len()).map(|_| false).collect();
        let tree = rstar::RTree::bulk_load(population.clone().into_iter().enumerate().map(|(point,idx)| RTreeItem::new(idx,point)).collect());
        Model {
            eps,
            mpt: min_points,
            tree, classification, visited, population,
        }
    }

    fn expand_cluster(
        classification: &mut [Classification],
        visited: &mut [bool],
        tree: &rstar::RTree<RTreeItem<T,P>>,
        population: &[P],
        eps: T,
        mpt: usize,
        index: usize,
        neighbors: &[usize],
        cluster: usize,
    )
    {
        classification[index] = Core(cluster);
        for &n_idx in neighbors {
            // Have we previously visited this point?
            let v = visited[n_idx];
            // n_idx is at least an edge point
            if classification[n_idx] == Noise {
                classification[n_idx] = Edge(cluster);
            }

            if !v {
                visited[n_idx] = true;
                // What about neighbors of this neighbor? Are they close enough to add into
                // the current cluster? If so, recurse and add them.
                let nn = Self::range_query_local(&population[n_idx], tree, eps);
                if nn.len() >= mpt {
                    // n_idx is a core point, we can reach at least min_points neighbors
                    Self::expand_cluster(
                        classification,
                        visited,
                        tree,
                        population,
                        eps,
                        mpt,
                        n_idx,
                        &nn,
                        cluster) 
                }
            }
        }
    }

    fn range_query_local(
        point: &P,
        tree: &rstar::RTree<RTreeItem<T,P>>,
        eps: T,
    ) -> Vec<usize> {
        let eps: T = T::normalize(eps.to_calc().powi(2));
        tree.locate_within_distance(*point, eps)
            .map(|ptr| ptr.get_idx())
            .collect()
    }

    #[cfg(test)]
    fn range_query2(&self, point: P) -> Vec<usize> {
        Self::range_query_local(&point, &self.tree, self.eps)
    }

    /*
    #[inline]
    fn range_query<A,C>(&self, sample: &A, population: &C) -> Vec<usize>
    where
        for<'a> &'a A: IntoIterator<Item=&'a T>,
        for<'b> &'b C: IntoIterator<Item=&'b P>,
        A: Point<T>,
        C: Population<P,T>,
    {
        population
            .into_iter()
            .enumerate()
            .filter(|(_, pt)| <A as Point<T>>::distance::<P,T>(sample, *pt) < self.eps)
            .map(|(idx, _)| idx)
            .collect()
    }
    */

    /// Run the DBSCAN algorithm on a given population of datapoints.
    ///
    /// A vector of [`Classification`] enums is returned, where each element
    /// corresponds to a row in the input matrix.
    ///
    /// # Arguments
    /// * `population` - a matrix of datapoints, organized by rows
    ///
    /// # Example
    ///
    /// ```rust
    /// use dbscan::Classification::*;
    /// use dbscan::Model;
    ///
    /// let inputs: Vec<[f64;2]> = vec![
    ///     [1.5, 2.2],
    ///     [1.0, 1.1],
    ///     [1.2, 1.4],
    ///     [0.8, 1.0],
    ///     [3.7, 4.0],
    ///     [3.9, 3.9],
    ///     [3.6, 4.1],
    ///     [10.0, 10.0],
    /// ];
    /// let mut model = Model::new(1.0, 3, inputs);
    /// let output = model.run();
    /// assert_eq!(
    ///     output,
    ///     vec![
    ///         Edge(0),
    ///         Core(0),
    ///         Core(0),
    ///         Core(0),
    ///         Core(1),
    ///         Core(1),
    ///         Core(1),
    ///         Noise
    ///     ]
    /// );
    /// ```
    pub fn run(&mut self) -> Vec<Classification> {
        let mut cluster = 0;
        for (idx, sample) in self.tree.iter().map(|ptr| (ptr.get_idx(),ptr.clone())) {
            let v = self.visited[idx];
            if !v {
                self.visited[idx] = true;
                let n = Self::range_query_local(&sample.point, &self.tree, self.eps);
                if n.len() >= self.mpt {
                    Self::expand_cluster(
                        &mut self.classification,
                        &mut self.visited,
                        &self.tree,
                        &self.population,
                        self.eps,
                        self.mpt,
                        idx,
                        &n,
                        cluster);
                    cluster += 1;
                }
            }
        }
        self.classification.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cluster() {
        let inputs: Vec<[f64;2]> = vec![
            [1.5, 2.2],
            [1.0, 1.1],
            [1.2, 1.4],
            [0.8, 1.0],
            [3.7, 4.0],
            [3.9, 3.9],
            [3.6, 4.1],
            [10.0, 10.0],
        ];
        let mut model = Model::new(1.0, 3, inputs);
        let output = model.run();
        assert_eq!(
            output,
            vec![
                Edge(0),
                Core(0),
                Core(0),
                Core(0),
                Core(1),
                Core(1),
                Core(1),
                Noise
            ]
        );
    }

    #[test]
    fn cluster_edge() {
        let inputs: Vec<[f64;5]> = vec![
            [
                0.3311755015020835,
                0.20474852214361858,
                0.21050489388506638,
                0.23040992344219402,
                0.023161159027037505,
            ],
            [
                0.5112445458548497,
                0.1898442816540571,
                0.11674072294944157,
                0.14853288499259437,
                0.03363756454905728,
            ],
            [
                0.581134172697341,
                0.15084733646825743,
                0.09997992993087741,
                0.13580335513916678,
                0.03223520576435743,
            ],
            [
                0.17210416043100868,
                0.3403172702783598,
                0.18218098373740396,
                0.2616980943829193,
                0.04369949117030829,
            ],
        ];
        let mut model = Model::new(0.253110, 3,inputs);
        let output = model.run();
        assert_eq!(output, vec![Core(0), Core(0), Edge(0), Edge(0)]);
    }

    #[test]
    fn range_query() {
        let inputs: Vec<[f64;2]> = vec![[1.0, 1.0], [1.1, 1.9], [3.0, 3.0]];
        let model = Model::new(1.0, 3, inputs);
        let point: [f64;2] = [1.0,1.0];
        let neighbours = model.range_query2(point);
        assert!(neighbours.len() == 2);
    }

    #[test]
    fn range_query_small_eps() {
        let inputs: Vec<[f64;2]> = vec![[1.0, 1.0], [1.1, 1.9], [3.0, 3.0]];
        let model = Model::new(0.01, 3, inputs);
        let point: [f64;2] = [1.0,1.0];
        let neighbours = model.range_query2(point);
        assert!(neighbours.len() == 1);
    }
}
