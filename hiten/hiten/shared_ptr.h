/*started shared ptr class, not used yet*/
// #pragma once

// #include <atomic>
// #include <memory>
// #include <type_traits>
// #include <iostream>

// // use this as a friend class later
// namespace pybind11 {
// template <typename, typename...>
// class class_;
// }

// namespace hiten {

// class shared_ptr_target {
//     mutable std::atomic<uint32_t> refcount;

//     template <class tgt>
//     friend class shared_ptr;
    
//     protected:
//         virtual ~shared_ptr_target() {}

//         constexpr shared_ptr_target() noexcept : refcount(0) {}

//         // disallow assignment or construction using other thing
//         shared_ptr_target(shared_ptr_target&&) noexcept : shared_ptr_target() {}

//         shared_ptr_target& operator=(shared_ptr_target&&) noexcept {
//             return *this;
//         }

//         shared_ptr_target(const shared_ptr_target&) noexcept : shared_ptr_target() {}

//         shared_ptr_target& operator=(const shared_ptr_target&) noexcept {
//             return *this;
//         }

//     private:
//         virtual void release_resources() {}

//         uint32_t refcount(std::memory_order order = std::memory_order_relaxed) const {
//             return refcount.load(order);
//         }

//         uint32_t refcount_incr(uint32_t incr) {
//             refcount.fetch_add(incr, std::memory_order_relaxed) + incr;
//         }

//         uint32_t refcount_decr(uint32_t decr) {
//             refcount.fetch_sub(decr, std::memory_order_relaxed) - decr;
//         }
// }

// template <class TTarget>
// class shared_ptr final {
//     private:
//         TTarget* target_;

//         template <class TT2>
//         friend class shared_ptr;

//         // Require for pybind https://pybind11.readthedocs.io/en/stable/advanced/smart_ptrs.html#custom-smart-pointers
//         template <typename, typename...>
//         friend class pybind11::class_;

//         void retain_() {
//             if (target != nullptr) {
//                 target_->refcount_incr(1);
//             }
//         }

//         void reset_() {
//             if (target != nullptr) {
//                 if (target->refcount.load(std::memory_order_acquire) == 1) {
//                     target->refcount.store(0, std::memory_order_relaxed);
//                     delete target_; // release resources
//                     return;
//                 }
//             }
//         }
//     public:
//         using element_type = TTarget;
//         shared_ptr() noexcept : target_(nullptr) {}

// }
// } // namespace hiten