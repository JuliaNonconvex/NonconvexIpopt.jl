@testitem "JET" begin
    using JET, NonconvexIpopt
    # The `symbolic` option exercises NonconvexCore's Symbolics extension, so
    # load it for JET to see the `symbolify` methods.
    using Symbolics
    JET.test_package(NonconvexIpopt; target_modules=(NonconvexIpopt,))
end
