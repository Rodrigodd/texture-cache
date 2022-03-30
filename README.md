# Texture Cache

A LRU texture cache for caching many small textures in a single big texture,
which is stored in GPU. This is used to cache textures that are rendered at
runtime and change constantly, like glyph text rendering.

This is basically a generic implementation of
[glyph_brush_draw_cache](https://github.com/alexheretic/glyph-brush/tree/master/draw-cache),
excluding the rendering part and letting you to hook your own rendering.

## Usage

Create a `LruTextureCache` and add rects by passing mutable slice of
`RectEntry` to the method `cache_rects`. A stored `Rect` can be queried from
the cache by passing it `key` to the method `get_rect`. `Rect` and `RectEntry`
can contain arbitrary data that could be useful for rendering from/to the
texture cache.

After passing the slice to `cache_rects`, it will be reorder so that it start
with every rect that was added to the cache.

```rust
use texture_cache::{LruTextureCache, RectEntry};

let mut rects = vec![RectEntry { width: 20, height: 20, key: "my_rect", value: (), entry_data: ()}];
let mut cache = LruTextureCache::new(256, 256);
let result = cache.cache_rects(&mut rects)?;

for rect in rects[0..result.len()] {
    let cached_rect = cache.get_rect(&rect.key);
    // Draw the rect to the texture cache...
}
```

## License

Licensed under either of

 * Apache License, Version 2.0, ([LICENSE-APACHE](LICENSE-APACHE) or
   http://www.apache.org/licenses/LICENSE-2.0)
 * MIT license ([LICENSE-MIT](LICENSE-MIT) or
   http://opensource.org/licenses/MIT)

at your option.

### Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in the work by you, as defined in the Apache-2.0 license, shall be
dual licensed as above, without any additional terms or conditions.
