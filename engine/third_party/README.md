# Vendored third-party headers

Header-only libraries used by the engine. Vendored (rather than fetched at build
time) so the project builds with a plain `cmake --build` and no network.

| library | version | license | used for |
|---|---|---|---|
| [RapidJSON](https://github.com/Tencent/rapidjson) | master (2024) | MIT (`rapidjson/LICENSE`) | config.json, vocab.json, the safetensors header, the shard index |
| [CLI11](https://github.com/CLIUtils/CLI11) | 2.6.2 (single header) | BSD-3-Clause (header banner) | command-line / sub-command parsing in `main.cpp` |
| [CUTLASS / CuTe](https://github.com/NVIDIA/cutlass) | 4.2.0 | BSD-3-Clause (`cutlass/LICENSE`) | CuTe tensor/layout algebra in `src/attn_cute.cu` |

## Updating CUTLASS

The headers under `cutlass/include/` are the header tree from the
`nvidia-cutlass` PyPI wheel (`cutlass_library/source/include`). To bump the version:

```sh
pip download nvidia-cutlass==<ver> --no-deps -d /tmp/cutlass-whl
unzip -o /tmp/cutlass-whl/nvidia_cutlass-*.whl -d /tmp/cutlass-whl/x
rm -rf cutlass/include && cp -r /tmp/cutlass-whl/x/cutlass_library/source/include cutlass/include
```

Then commit the result.
