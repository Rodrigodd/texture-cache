//! A LRU texture cache for caching many small textures in a single big texture, which is stored in
//! GPU. This is used to cache textures that are rendered at runtime and change constantly, like
//! glyph text rendering.
//!
//! ## Usage
//!
//! Create a [`LruTextureCache`] and add rects by passing mutable slice of [`RectEntry`] to the
//! method [`cache_rects`](LruTextureCache::cache_rects). A stored [`Rect`] can be queried from the
//! cache by passing it `key` to the method `get_rect`. `Rect` and `RectEntry` can contain
//! arbitrary data that could be useful for rendering from/to the texture cache.
//!
//! After passing the slice to `cache_rects`, it will be reorder so that it start with every rect
//! that was added to the cache. See [`LruTextureCache::cache_rects`] for details.
//!
//! ## Example
//!
//! ```rust
//! # fn main() -> Result<(), texture_cache::CacheErr> {
//! use texture_cache::{LruTextureCache, RectEntry};
//!
//! let mut rects = vec![RectEntry {
//!     width: 20,
//!     height: 20,
//!     key: "my_rect",
//!     value: (),
//!     entry_data: (),
//! }];
//! let mut cache = LruTextureCache::new(256, 256);
//! let result = cache.cache_rects(&mut rects)?;
//!
//! for rect in &rects[0..result.len()] {
//!     let cached_rect = cache.get_rect(&rect.key);
//!     // Draw the rect to the texture
//! }
//! # Ok(())
//! # }
//! ```

#![warn(missing_docs)]

use std::collections::HashMap;
use std::hash::Hash;

#[cfg(test)]
mod test {
    use std::fmt::Debug;

    use super::*;

    fn check_overlaps<K: Hash + Eq + Debug, V>(cache: &LruTextureCache<K, V>) {
        let rects = cache.rows.iter().flat_map(|x| x.rects.iter()).enumerate();
        let overlap = |a: &Rect<V>, b: &Rect<V>| {
            a.x < b.x + b.width
                && b.x < a.x + a.width
                && a.y < b.y + b.height
                && b.y < a.y + a.height
        };
        for (x, this) in rects.clone() {
            if this.x + this.width > cache.width || this.y + this.height > cache.height {
                panic!("rect overflow");
            }
            for (y, other) in rects.clone() {
                if x != y {
                    assert!(!overlap(this, other), "detected overlap");
                }
            }
        }

        let mut values = cache.key_to_row.values().collect::<Vec<_>>();
        let len = values.len();
        values.sort();
        println!("{:?}", values);
        values.dedup();
        assert_eq!(values.len(), len, "{:?}", cache.key_to_row);
    }

    #[test]
    fn too_big() {
        let mut cache = LruTextureCache::new(100, 100);
        assert!(cache.add_rect(150, 50, 0, ()) == AddRectResult::TooLarge);
        assert!(cache.add_rect(50, 150, 1, ()) == AddRectResult::TooLarge);
        assert!(cache.add_rect(101, 0, 0, ()) == AddRectResult::TooLarge);
    }

    #[test]
    fn row_count() {
        let mut cache = LruTextureCache::new(100, 100);
        let mut counter = -1;
        let mut rects = move || {
            (0..10)
                .map(|_| {
                    counter += 1;
                    RectEntry {
                        width: 10,
                        height: 10,
                        key: counter,
                        value: (),
                        entry_data: (),
                    }
                })
                .collect::<Vec<_>>()
        };

        for i in 0..10 {
            let result = cache.cache_rects(&mut rects());
            assert_eq!(result, Ok(Cached::Added(10)));
            check_overlaps(&cache);
            assert_eq!(cache.rows.len(), i + 1);
        }

        let result = cache.cache_rects(&mut rects());
        assert_eq!(result, Ok(Cached::Changed(10)));
        check_overlaps(&cache);
        assert_eq!(cache.rows.len(), 10);

        for i in 0..10 {
            assert!(!cache.contains(&i));
        }
        for i in 10..110 {
            assert!(cache.contains(&i));
        }
    }

    #[test]
    fn random_sample() {
        use rand::prelude::*;
        let seed = thread_rng().gen::<u64>();
        println!("set seed {:016x}", seed);
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let mut gen = || rng.gen_range(1..=10) + rng.gen_range(1..=10);
        let rects: Vec<_> = (0..1000)
            .map(|x| RectEntry {
                width: gen(),
                height: gen(),
                key: x,
                value: (),
                entry_data: (),
            })
            .collect();
        let mut cache = LruTextureCache::new(100, 100);
        for i in 0..200 {
            println!("sample number {}", i);
            let size = rng.gen_range(25..100);
            let mut sample = rects.iter().cloned().choose_multiple(&mut rng, size);
            if cache.cache_rects(&mut sample).is_ok() {
                for rect in sample.iter() {
                    assert!(cache.contains(&rect.key));
                }
            }
            check_overlaps(&cache);
        }
    }

    #[test]
    #[ignore]
    /// This test is only used repeat the `random_sample` test, but with a predetermined seed.
    fn seed_sample() {
        use rand::prelude::*;
        let seed = 0xe850b0f0dae9bbd6;
        println!("set seed {:016x}", seed);
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let mut gen = || rng.gen_range(1..=10) + rng.gen_range(1..=10);
        let rects: Vec<_> = (0..1000)
            .map(|x| RectEntry {
                width: gen(),
                height: gen(),
                key: x,
                value: (),
                entry_data: (),
            })
            .collect();
        let mut cache = LruTextureCache::new(100, 100);
        for i in 0..2 {
            println!("sample number {}", i);
            let size = rng.gen_range(25..100);
            let mut sample = rects.iter().cloned().choose_multiple(&mut rng, size);
            if cache.cache_rects(&mut sample).is_ok() {
                for rect in sample.iter() {
                    assert!(cache.contains(&rect.key));
                }
            }
            check_overlaps(&cache);
        }
    }

    #[test]
    fn multiple_size() {
        let mut cache = LruTextureCache::new(100, 100);
        let i = std::sync::atomic::AtomicU64::new(0);
        let count = move || i.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let rects = (0..11)
            .map(|_| RectEntry {
                width: 15,
                height: 15,
                key: count(),
                value: (),
                entry_data: (),
            })
            .chain((0..3).map(|_| RectEntry {
                width: 10,
                height: 10,
                key: count(),
                value: (),
                entry_data: (),
            }));
        let mut rects: Vec<_> = rects.collect();
        cache.cache_rects(&mut rects).unwrap();
        println!(
            "{:?}",
            cache.rows.iter().map(|x| &x.rects).collect::<Vec<_>>()
        );
    }
}

#[derive(Clone, Debug)]
/// The stored Rect. Contains its dimensions, position and the associated value.
pub struct Rect<V> {
    /// The horizontal position, in pixels, of the top left corner of the rect in the texture. 0 is
    /// the left edge of the texture.
    pub x: u32,
    /// The vertical position, in pixels, of the top left corner of the rect in the texture. 0 is
    /// the top edge of the texture.
    pub y: u32,
    /// The width of the rect, in pixels.
    pub width: u32,
    /// The height of the rect, in pixels.
    pub height: u32,
    /// The value associated with this rect.
    pub value: V,
}

struct Row<V> {
    /// The age of the row since the last time it was used to cache a rect.  Increase by one for
    /// ever call to `cache_rects`. Reset to 0 when store a rect from `cache_rects`.
    age: u8,
    /// The position of the top of the row
    top: u32,
    /// The height of the row
    height: u32,
    /// The space that is not occupied to the right of the last stored rect.
    free_space: u32,
    /// The rects stored in this row
    pub rects: Vec<Rect<V>>,
}
impl<V> Row<V> {
    pub fn push_rect(&mut self, width: u32, height: u32, value: V) {
        let y = self.top;
        let x = self.rects.last().map_or(0, |x| x.x + x.width);
        let rect = Rect {
            x,
            y,
            width,
            height,
            value,
        };
        self.free_space -= rect.width;
        self.rects.push(rect);
    }

    fn len(&self) -> usize {
        self.rects.len()
    }
}

/// The entry of a rect to be cached.
#[derive(Clone, Hash, PartialEq, Eq)]
pub struct RectEntry<K: Hash + Eq + Clone, V: Clone, D> {
    /// The width of the rect to be cached.
    pub width: u32,
    /// The height of the rect to be cached.
    pub height: u32,
    /// The key that will be mapped to the cached rect.
    pub key: K,
    /// A value which will be associated with the cached rect.
    pub value: V,
    /// A value it will be associated with this rect entry. This is not stored in the cache, but it
    /// is used to do operations with this entry right after adding it.
    pub entry_data: D,
}

type RowIndex = u32;
type RectIndex = u32;

/// Successful method of caching of the queue, returned from `cache_rects`.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Cached {
    /// Added some rects into the cache without affecting any of the already cached rects from
    /// previous queues. Contains the number of rects that was added to the cache.
    Added(usize),
    /// Added some rects into the cache, but removed some glyphs from previous queues. Contains the
    /// number of rects that was added to the cache.
    Changed(usize),
    /// Added all rects into the cache, by clearing all rects from previous queues. Contains the
    /// number of rects contained in the cache.
    Cleared(usize),
}
impl Cached {
    /// Return the number of rects that was added to the cached.
    ///
    /// This includes rects that was reordered, in the cause of [Cached::Cleared].
    pub fn len(&self) -> usize {
        match *self {
            Self::Added(len) => len,
            Self::Changed(len) => len,
            Self::Cleared(len) => len,
        }
    }
}

/// Reason of cache failure.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum CacheErr {
    /// At least one of the queued rects is too big to fit into the cache by itself. Contains the
    /// number of rects that was added to the cache.
    RectTooLarge(usize),
    /// Not all of the requested glyphs fit into the cache, even after clearing all previous cached
    /// rects. Contains the number of rects that was added to the cache.
    DontFit(usize),
}
impl CacheErr {
    /// Return the number of rects that was added to the cached.
    pub fn len(&self) -> usize {
        match *self {
            Self::RectTooLarge(len) => len,
            Self::DontFit(len) => len,
        }
    }
}

#[derive(PartialEq, Eq, Debug)]
enum AddRectResult {
    Added,
    ClearedRows,
    TooLarge,
    NoRowToFit,
}

/// A LRU texture cache to cache rects in a texture.
///
///## Algorithm
/// This works by dividing the texture in rows. When adding a rect it check if it fit in any of the
/// existing rows. If not, a new row is created, with the height of the added rect. If there is no
/// space for new rows, a range of unused rows with a height that fit the rect is remove, and a new
/// row that fit the rect is added. If no row is find, the entire cache is cleared.
///
/// When adding multiple rects, all rows that contains any of the already added rects are marked as
/// used, and the remain are the unused ones.
pub struct LruTextureCache<K: Hash + Eq, V> {
    /// The width of the texture.
    width: u32,
    /// The height of the texture.
    height: u32,
    /// Alllocated rows in the cache.
    rows: Vec<Row<V>>,
    /// A map from a rect key to the row and index it is stored.
    key_to_row: HashMap<K, (RowIndex, RectIndex)>,
    /// Free space below the bottom row.
    free_space: u32,
}
impl<K: Hash + Eq + Copy + 'static, V: Clone> LruTextureCache<K, V> {
    /// Create a new empty cache, with the given width and height.
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            rows: Default::default(),
            key_to_row: Default::default(),
            free_space: height,
        }
    }
    /// The width of the cache
    pub fn width(&self) -> u32 {
        self.width
    }

    /// The height of the cache
    pub fn height(&self) -> u32 {
        self.height
    }

    /// The height of the occupied area of the cache.
    ///
    /// Cached rects are placed at the top, and grows to bottom, this is the position of the bottom
    /// of the lowest rect.
    pub fn min_height(&self) -> u32 {
        self.height - self.free_space
    }

    /// The number of rects currently in the cache
    pub fn len(&self) -> usize {
        self.key_to_row.len()
    }

    /// Return true if there is no rect in the cache.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Clears the cache, leaving it as it if were new.
    pub fn clear(&mut self) {
        self.free_space = self.height;
        self.rows.clear();
        self.key_to_row.clear();
    }

    /// Return true if there is a cached rect associated with the given Key. Otherwise, return
    /// false.
    pub fn contains(&self, key: &K) -> bool {
        self.key_to_row.contains_key(key)
    }

    /// partition the given slice in uncached and cached, and return the index of partition.
    /// Uncached rects will add a dummy value to the `key_to_row` map.
    fn partition_cached<D>(&mut self, rects: &mut [RectEntry<K, V, D>]) -> usize {
        fn partition<T, F: FnMut(&T) -> bool>(slice: &mut [T], mut f: F) -> usize {
            let mut l = 0;
            for i in 0..slice.len() {
                if f(&slice[i]) {
                    slice.swap(i, l);
                    l += 1;
                }
            }
            l
        }
        partition(rects, |x| {
            if !self.key_to_row.contains_key(&x.key) {
                // insert with a dummy value, this is replace later when the rect is added.
                self.key_to_row.insert(x.key, (!0, !0));
                true
            } else {
                false
            }
        })
    }

    /// Cache the given slice of rects.
    ///
    /// The given slice of rects is reorder so that it start with every rect that was added to the
    /// cache. This way you can iterate over them by subslicing `rects`  (`rects[0..len]`, where
    /// `len` is [`Cached::len`] or [`CacheErr::len`]) and draw them to the texture cache.
    ///
    /// If return `Ok`, it is guaranteed that each one of the given rects will be present in the
    /// cache after this call, but any rects from previous calls could have been removed.
    #[must_use]
    pub fn cache_rects<D>(&mut self, rects: &mut [RectEntry<K, V, D>]) -> Result<Cached, CacheErr> {
        // Partition the vector in uncached and cached. Be aware of the dummy values.
        let s = self.partition_cached(rects);
        // sort uncached by decresing height
        rects[..s].sort_unstable_by_key(|x| !x.height);

        // Update the use age of the rows for this batch. Uncached rects will mark the rows when added.

        for row in &mut self.rows {
            row.age = row.age.saturating_add(1);
        }
        for rect in &rects[s..] {
            let &(row, _) = self.key_to_row.get(&rect.key).unwrap();
            // be careful with dummy values
            if row == !0 {
                continue;
            }
            self.rows[row as usize].age = 0;
        }

        // try add all rects to the cache

        enum Sucess {
            Okay,
            Change,
            Fail,
        }
        use AddRectResult::*;
        use Sucess::*;
        let mut sucess = Okay;
        for (r, rect) in rects[..s].iter().enumerate() {
            match self.add_rect(rect.width, rect.height, rect.key, rect.value.clone()) {
                Added => {}
                ClearedRows => sucess = Change,
                NoRowToFit => {
                    sucess = Fail;
                    break;
                }
                TooLarge => {
                    // clear dummy key_to_row values
                    for rect in &rects[r..s] {
                        self.key_to_row.remove(&rect.key);
                    }
                    return Err(CacheErr::RectTooLarge(r));
                }
            }
        }

        match sucess {
            Okay => Ok(Cached::Added(s)),
            Change => Ok(Cached::Changed(s)),

            // if the rects don't fit in the cache, clear everthing and try again
            Fail => {
                self.clear();
                // partition the vector in uncached and cached
                let s = self.partition_cached(rects);
                // sort uncached by decresing height
                rects[..s].sort_unstable_by_key(|x| !x.height);
                for (r, rect) in rects[..s].iter().enumerate() {
                    match self.add_rect(rect.width, rect.height, rect.key, rect.value.clone()) {
                        Added => {}
                        ClearedRows | TooLarge => unreachable!(),
                        NoRowToFit => {
                            // clear dummy key_to_row values
                            for rect in &rects[r..s] {
                                self.key_to_row.remove(&rect.key);
                            }
                            return Err(CacheErr::DontFit(r));
                        }
                    }
                }
                Ok(Cached::Cleared(s))
            }
        }
    }

    /// Return the rect where the texture with the given key is stored in the texture.
    pub fn get_rect(&self, key: &K) -> Option<Rect<V>> {
        let (row, index) = *self.key_to_row.get(&key)?;
        Some(self.rows[row as usize].rects[index as usize].clone())
    }

    /// Add a rect to a row in the cache, and mark the row as used. This can deallocate any row
    /// that is not marked as used.
    fn add_rect(&mut self, width: u32, height: u32, key: K, value: V) -> AddRectResult {
        if width > self.width || height > self.height {
            return AddRectResult::TooLarge;
        }
        // put in the first rows that fits
        for (r, row) in self.rows.iter_mut().enumerate() {
            if row.height >= height && row.free_space >= width {
                // reborrow because of lifetime issues
                let row = &mut self.rows[r];
                row.age = 0;
                self.key_to_row
                    .insert(key, (r as RowIndex, row.len() as RectIndex));
                row.push_rect(width, height, value);
                return AddRectResult::Added;
            }
        }

        // if don't fit in any row, add a new one
        if self.free_space >= height {
            self.free_space -= height;
            let mut row = Row {
                age: 0,
                top: self.rows.last().map_or(0, |x| x.top + x.height),
                height,
                free_space: self.width,
                rects: Vec::new(),
            };
            self.key_to_row
                .insert(key, (self.rows.len() as RowIndex, 0 as RectIndex));
            row.push_rect(width, height, value);
            self.rows.push(row);
            return AddRectResult::Added;
        }

        // if there is no space for new rows, clear unused ones to fit the new rect

        // find the best range of consecutive unused rows that fit the rect.
        // older is better.
        let mut possible_row = None;
        let mut best = 0;
        for r in 0..self.rows.len() {
            let mut rows_height = 0;
            // the age of the yonger row
            let mut age = !0;
            for o in r..self.rows.len() {
                let row = &self.rows[o];
                if row.age == 0 {
                    break;
                }
                if row.age < age {
                    if row.age <= best {
                        break;
                    }
                    age = row.age;
                }
                rows_height += row.height;
                if rows_height >= height {
                    possible_row = Some((r..o + 1, rows_height));
                    best = age;
                    break;
                }
            }
        }
        if let Some((range, row_height)) = possible_row {
            // TODO: don't clear all three rows, keep the rects from one of them.
            let top = self.rows[range.start as usize].top;
            let mut new_row = Row {
                age: 0,
                top,
                height,
                free_space: self.width,
                rects: Vec::new(),
            };
            new_row.push_rect(width, height, value);
            let add_len = if row_height == height {
                self.rows.splice(range.clone(), std::iter::once(new_row));
                1
            } else {
                // create a second row, to fill the gap
                let split_row = Row {
                    age: !0,
                    top: top + height,
                    height: row_height - height,
                    free_space: self.width,
                    rects: Vec::new(),
                };
                self.rows
                    .splice(range.clone(), IntoIterator::into_iter([new_row, split_row]));
                2
            };

            // remove and shift rows from key_to_row hash map
            self.key_to_row.retain(|_, (row, _)| {
                if (*row as usize) < range.start {
                    true
                } else if (*row as usize) >= range.end {
                    if *row != !0 {
                        *row = *row + add_len - range.len() as u32;
                    }
                    true
                } else {
                    false
                }
            });
            self.key_to_row
                .insert(key, (range.start as RowIndex, 0 as RectIndex));
            return AddRectResult::ClearedRows;
        }

        // if there is no row, the operation failed.
        AddRectResult::NoRowToFit
    }
}
