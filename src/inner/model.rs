use std::{
    num::NonZeroUsize,
};

use crate::{
    Classification,
    inner::{
        data_scalar::{DataScalar},
        point::{Point},
        state_tracker::{StateTracker,StateCell},
        item::{RTreeItem},
    },
};


pub fn build_clusters<
    'input,
    'a,
    S: DataScalar,
    P: Point<S>,
    const MIN_SIZE: usize
>(
    state: &mut StateTracker<'input,S,P>,
    tree: &'a rstar::RTree<RTreeItem<'input,S,P>>,
    eps: S
) {
    let mut cluster = 0usize;
    for idx in state.range_iter() {
        if state[idx].was_visisted() {
            continue;
        }
        match range_query::<S,P,MIN_SIZE>(state[idx].get_point(), tree, eps.clone()) {
            Option::None => { },
            Option::Some(iter) => {
                state[idx].set_class(Classification::Core(cluster));
                expand_cluster::<S,P,MIN_SIZE>(state, tree, eps.clone(), iter, cluster);
                cluster += 1;
            }
        };
    }
}

fn expand_cluster<
    'input,
    'a,
    S: DataScalar,
    P: Point<S>,
    const MIN_SIZE: usize,
>(
    state: &mut StateTracker<'input,S,P>,
    tree: &'a rstar::RTree<RTreeItem<'input,S,P>>, 
    eps: S,
    neighbors: Box<dyn Iterator<Item=NonZeroUsize>+'a>,
    cluster: usize,
) {
    let mut iter: Box<dyn Iterator<Item=NonZeroUsize> +'a> = neighbors;
    loop {
        let idx = match iter.next() {
            Option::None => {
                return;
            }
            Option::Some(idx) => {
                if state[idx].is_noise() {
                    state[idx].set_class(Classification::Edge(cluster));
                }
                if state[idx].was_visisted() {
                    continue;
                }
                idx
            }
        };
        state[idx].set_visisted();
        match range_query::<S,P,MIN_SIZE>(state[idx].get_point(), tree, eps.clone()) {
            Option::None => {
                continue;
            }
            Option::Some(inner) => {
                state[idx].set_class(Classification::Core(cluster));
                iter = Box::new(iter.chain(inner));
            }
        };
    }
}

pub fn range_query<
    'input,
    'a,
    S: DataScalar,
    P: Point<S>,
    const MIN_SIZE: usize
>(
    point: &'a P,
    tree: &'a rstar::RTree<RTreeItem<'input,S,P>>,
    eps: S,
) -> Option<Box<dyn Iterator<Item=NonZeroUsize> + 'a>> {
    let mut items: [Option<NonZeroUsize>;MIN_SIZE] = [None;MIN_SIZE];
    let mut iter = tree.locate_within_distance(*point, eps)
        .map(|p| p.get_idx());
    for i in 0..MIN_SIZE {
        match iter.next() {
            Option::Some(x) => {
                items[i] = Some(x);
            },
            Option::None => {
                return None;
            }
        };
    }
    Some(Box::new(items.into_iter().filter_map(|x| x).chain(iter)))
}





