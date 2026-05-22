
"""
    build_sindy_basis(state_vars, input_vars, function_library) -> Basis

Constructs a `DataDrivenDiffEq.Basis` from symbolic state and control variables
generated from the provided name vectors, using `function_library` to produce
the candidate terms.
"""
function build_sindy_basis(
    state_vars::Vector{String},
    input_vars::Vector{String},
    function_library
)::Basis
    @variables τ

    all_names = vcat(state_vars, input_vars)
    sym_vars  = [only(@variables $(Symbol(v))(τ)) for v in all_names]

    n_state = length(state_vars)
    u_vars  = sym_vars[1:n_state]
    c_vars  = sym_vars[n_state+1:end]

    return Basis(function_library(sym_vars...), u_vars; controls = c_vars, iv = τ)
end


