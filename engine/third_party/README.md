# Vendored third-party headers

Header-only libraries used for JSON and command-line parsing — utility code, not
part of the from-scratch inference engine. Vendored (rather than fetched at build
time) so the project still builds with a plain `cmake --build` and no network.

| library | version | license | used for |
|---|---|---|---|
| [RapidJSON](https://github.com/Tencent/rapidjson) | master (2024) | MIT (`rapidjson/LICENSE`) | config.json, vocab.json, the safetensors header, the shard index |
| [CLI11](https://github.com/CLIUtils/CLI11) | 2.6.2 (single header) | BSD-3-Clause (header banner) | command-line / sub-command parsing in `main.cpp` |
