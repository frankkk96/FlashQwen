// Engine startup state, shared between the loader thread (writes) and the gRPC handler thread
// (reads, to answer GetStatus). Deliberately grpc-free — the service translates a Snapshot into the
// proto StatusResponse, keeping this header dependency-light like errors.hpp.
#pragma once
#include <atomic>
#include <mutex>
#include <string>
#include <utility>

enum class LoadState { Loading, Ready, Failed };

class StartupStatus {
public:
    // Begin a new phase (e.g. "loading weights"); total is the unit count if countable, else 0.
    void set_phase(std::string phase, int total = 0) {
        std::lock_guard<std::mutex> lk(mu_);
        phase_ = std::move(phase);
        done_ = 0;
        total_ = total;
    }
    // Report progress within the current phase.
    void advance(int done, int total) {
        std::lock_guard<std::mutex> lk(mu_);
        done_ = done;
        total_ = total;
    }
    void mark_ready() { state_.store(LoadState::Ready); }
    void mark_failed(std::string message) {
        {
            std::lock_guard<std::mutex> lk(mu_);
            message_ = std::move(message);
        }
        state_.store(LoadState::Failed);
    }

    struct Snapshot {
        LoadState state;
        std::string phase;
        int done, total;
        std::string message;
    };
    Snapshot snapshot() {
        std::lock_guard<std::mutex> lk(mu_);
        return {state_.load(), phase_, done_, total_, message_};
    }

private:
    std::atomic<LoadState> state_{LoadState::Loading};
    std::mutex mu_;
    std::string phase_ = "starting";
    int done_ = 0, total_ = 0;
    std::string message_;
};
