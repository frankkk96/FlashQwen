// Local inference server: a persistent process that loads the model once and serves many
// requests concurrently through the continuous-batching Scheduler. The Go OpenAI gateway is the
// only client. Transport is a Unix-domain socket speaking newline-delimited JSON:
//
//   request  (gateway -> engine), one line:
//     {"prompt":"<rendered chat-template text>","max_tokens":256,"temperature":0.7,"top_p":0.95}
//   response (engine -> gateway), a stream of lines, terminated by a done line:
//     {"delta":"Hello","done":false}
//     {"delta":" world","done":false}
//     {"done":true,"finish_reason":"stop","prompt_tokens":12,"completion_tokens":34}
//
// One request per connection. The gateway tokenises nothing — it sends the fully rendered prompt
// text (chat template + tools already applied) and the engine tokenises / detokenises.
#pragma once
#include "model.hpp"
#include "tokenizer.hpp"
#include <string>
#include <random>

int run_server(Model& model, const Tokenizer& tok, const std::string& socket_path,
               int n_slots, std::mt19937& rng);
