using JET, NonconvexIpopt

# Run JET analysis on the package.
JET.test_package(NonconvexIpopt; target_modules=(NonconvexIpopt,))
