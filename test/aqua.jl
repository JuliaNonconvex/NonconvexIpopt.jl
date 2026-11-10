@testitem "Aqua" begin
    using Aqua, NonconvexIpopt
    Aqua.test_all(NonconvexIpopt)
end
