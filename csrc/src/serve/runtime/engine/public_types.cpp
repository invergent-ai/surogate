#include "api/types.h"

#include <utility>

namespace sinfer {

CancellationView::CancellationView(std::function<bool()> requested)
    : requested_(std::move(requested)) {}

bool CancellationView::requested() const { return requested_ && requested_(); }

} // namespace sinfer
