#JuMP model, Ipopt solver
using JuMP, Ipopt, Plots

sys = Model(optimizer_with_attributes(Ipopt.Optimizer,"print_level"=> 5))
set_optimizer_attribute(sys,"tol",1e-6)
set_optimizer_attribute(sys,"max_iter",1000)

# Parameters
w = 0.8
x0 = 0 
y0 = 0 
θ0 = π/7
xf = 4
yf = 7
θf =-π/2 
N = 100

# Bounds for variables

@variables(sys,begin
    x[1:N + 1]         
    y[1:N + 1]         
    θ[1:N + 1]         
    -1 ≤ u[1:N] ≤ 1
    0 ≤ Δt ≤ 1 
    end)

# Objective
@objective(sys, Min, Δt)

# Constraints 
@constraints(sys, begin
    x[1] == x0
    y[1] == y0
    θ[1] == θ0
    x[N + 1] == xf
    y[N + 1] == yf
    θ[N + 1] == θf
    end)

# Dynamics: Euler scheme
for j in 1:N
    @NLconstraint(sys, # x' = w + cos(theta)
        x[j+1] == x[j] + Δt * (w + cos(θ[j])))
    @NLconstraint(sys, # y' = sin(theta) 
        y[j+1] == y[j] + Δt * sin(θ[j]))
    @NLconstraint(sys, # theta' = u 
        θ[j+1] == θ[j] + Δt * u[j])
end


# Solves for the control and state
println("Solving...")
status = optimize!(sys)
println("Solver status : ",status)
x1 = value.(x)
y1 = value.(y)
θ1 = value.(θ)
u1 = value.(u)
println("Cost : " , objective_value(sys))
println("tf = ", value.(Δt) * N)

# Plots: states 
Δt1 = value.(Δt)
t = (0:N) * Δt1
x_plot = plot(t, x1; xlabel="t", ylabel="position x", legend=false, fmt=:png)
y_plot = plot(t, y1; xlabel="t", ylabel="position y", legend=false, fmt=:png)
θ_plot = plot(t, θ1; xlabel="t", ylabel="θ", legend=false, fmt=:png)
u_plot = plot(t[1:end-1], u1; xlabel="t", ylabel="control", legend=false, fmt=:png)
display(plot(x_plot, y_plot, θ_plot, u_plot; layout=(2,2)))

# Plots: trajectory 
step = 5
traj_plot = plot(x1, y1; c=:black, lw=3)
plot!(size=(600,600))
scatter!(traj_plot, x1[1:step:end], y1[1:step:end]; c=:red, legend=false)



# Point milieu 
sys = Model(optimizer_with_attributes(Ipopt.Optimizer, "print_level" => 0))
set_optimizer_attribute(sys, "tol", 1e-6)
set_optimizer_attribute(sys, "max_iter", 1000)

@variables(sys, begin
    x[1:N+1]
    y[1:N+1]
    θ[1:N+1]
    -1 <= u[1:N] <= 1
    0 <= Δt <= 1
end)

@objective(sys, Min, Δt)

@constraints(sys, begin
    x[1] == x0
    y[1] == y0
    θ[1] == θ0
    x[N+1] == xf
    y[N+1] == yf
    θ[N+1] == θf
end)

for j in 1:N
    θ_milieu = @NLexpression(sys, θ[j] + (Δt / 2) * u[j])

    @NLconstraint(sys, x[j+1] == x[j] + Δt * (w + cos(θ_milieu)))
    @NLconstraint(sys, y[j+1] == y[j] + Δt * sin(θ_milieu))
    @NLconstraint(sys, θ[j+1] == θ[j] + Δt * u[j])
end

optimize!(sys)
println("Statut : ", termination_status(sys))
@assert is_solved_and_feasible(sys) "Point milieu : Ipopt n'a pas convergé."

x1 = value.(x)
y1 = value.(y)
θ1 = value.(θ)
u1 = value.(u)
Δt1 = value(Δt)
tf_milieu = N * Δt1
t = (0:N) * Δt1

println("Δt = ", Δt1)
println("tf = ", tf_milieu)

x_plot = plot(t, x1; xlabel="t", ylabel="x", legend=false)
y_plot = plot(t, y1; xlabel="t", ylabel="y", legend=false)
θ_plot = plot(t, θ1; xlabel="t", ylabel="θ", legend=false)
u_plot = plot(t, [u1; u1[end]]; xlabel="t", ylabel="u", legend=false, seriestype=:steppost)
display(plot(x_plot, y_plot, θ_plot, u_plot; layout=(2, 2), plot_title="Point milieu"))
display(plot(x1, y1; xlabel="x", ylabel="y", title="Point milieu : trajectoire", legend=false, aspect_ratio=:equal))


# Trapeze
sys = Model(optimizer_with_attributes(Ipopt.Optimizer, "print_level" => 0))
set_optimizer_attribute(sys, "tol", 1e-6)
set_optimizer_attribute(sys, "max_iter", 1000)

@variables(sys, begin
    x[1:N+1]
    y[1:N+1]
    θ[1:N+1]
    -1 <= u[1:N] <= 1
    0 <= Δt <= 1
end)

@objective(sys, Min, Δt)

@constraints(sys, begin
    x[1] == x0
    y[1] == y0
    θ[1] == θ0
    x[N+1] == xf
    y[N+1] == yf
    θ[N+1] == θf
end)

for j in 1:N
    @NLconstraint(sys, x[j+1] == x[j] + (Δt / 2) * ((w + cos(θ[j])) + (w + cos(θ[j+1]))))
    @NLconstraint(sys, y[j+1] == y[j] + (Δt / 2) * (sin(θ[j]) + sin(θ[j+1])))
    @NLconstraint(sys, θ[j+1] == θ[j] + Δt * u[j])
end

optimize!(sys)
println("Statut : ", termination_status(sys))
@assert is_solved_and_feasible(sys) "Trapèze : Ipopt n'a pas convergé."

x1 = value.(x)
y1 = value.(y)
θ1 = value.(θ)
u1 = value.(u)
Δt1 = value(Δt)
tf_trapeze = N * Δt1
t = (0:N) * Δt1

println("Δt = ", Δt1)
println("tf = ", tf_trapeze)

x_plot = plot(t, x1; xlabel="t", ylabel="x", legend=false)
y_plot = plot(t, y1; xlabel="t", ylabel="y", legend=false)
θ_plot = plot(t, θ1; xlabel="t", ylabel="θ", legend=false)
u_plot = plot(t, [u1; u1[end]]; xlabel="t", ylabel="u", legend=false, seriestype=:steppost)
display(plot(x_plot, y_plot, θ_plot, u_plot; layout=(2, 2), plot_title="Trapèze"))
display(plot(x1, y1; xlabel="x", ylabel="y", title="Trapèze : trajectoire", legend=false, aspect_ratio=:equal))


# Runge Kutta 4 
sys = Model(optimizer_with_attributes(Ipopt.Optimizer, "print_level" => 0))
set_optimizer_attribute(sys, "tol", 1e-6)
set_optimizer_attribute(sys, "max_iter", 1000)

@variables(sys, begin
    x[1:N+1]
    y[1:N+1]
    θ[1:N+1]
    -1 <= u[1:N] <= 1
    0 <= Δt <= 1
end)

@objective(sys, Min, Δt)

@constraints(sys, begin
    x[1] == x0
    y[1] == y0
    θ[1] == θ0
    x[N+1] == xf
    y[N+1] == yf
    θ[N+1] == θf
end)

for j in 1:N
    # Première pente
    k1x = @NLexpression(sys, w + cos(θ[j]))
    k1y = @NLexpression(sys, sin(θ[j]))
    k1θ = u[j]

    # Deuxième pente : premier point au milieu du pas
    k2x = @NLexpression(sys, w + cos(θ[j] + (Δt / 2) * k1θ))
    k2y = @NLexpression(sys, sin(θ[j] + (Δt / 2) * k1θ))
    k2θ = u[j]

    # Troisième pente : second point au milieu du pas
    k3x = @NLexpression(sys, w + cos(θ[j] + (Δt / 2) * k2θ))
    k3y = @NLexpression(sys, sin(θ[j] + (Δt / 2) * k2θ))
    k3θ = u[j]

    # Quatrième pente : point à la fin du pas
    k4x = @NLexpression(sys, w + cos(θ[j] + Δt * k3θ))
    k4y = @NLexpression(sys, sin(θ[j] + Δt * k3θ))
    k4θ = u[j]

    # Moyenne pondérée des quatre pentes
    @NLconstraint(sys, x[j+1] == x[j] + (Δt / 6) * (k1x + 2*k2x + 2*k3x + k4x))
    @NLconstraint(sys, y[j+1] == y[j] + (Δt / 6) * (k1y + 2*k2y + 2*k3y + k4y))
    @NLconstraint(sys, θ[j+1] == θ[j] + (Δt / 6) * (k1θ + 2*k2θ + 2*k3θ + k4θ))
end

optimize!(sys)
println("Statut : ", termination_status(sys))
@assert is_solved_and_feasible(sys) "RK4 : Ipopt n'a pas convergé."

x1 = value.(x)
y1 = value.(y)
θ1 = value.(θ)
u1 = value.(u)
Δt1 = value(Δt)
tf_rk4 = N * Δt1
t = (0:N) * Δt1

println("Δt = ", Δt1)
println("tf = ", tf_rk4)

x_plot = plot(t, x1; xlabel="t", ylabel="x", legend=false)
y_plot = plot(t, y1; xlabel="t", ylabel="y", legend=false)
θ_plot = plot(t, θ1; xlabel="t", ylabel="θ", legend=false)
u_plot = plot(t, [u1; u1[end]]; xlabel="t", ylabel="u", legend=false, seriestype=:steppost)
display(plot(x_plot, y_plot, θ_plot, u_plot; layout=(2, 2), plot_title="RK4"))
display(plot(x1, y1; xlabel="x", ylabel="y", title="RK4 : trajectoire", legend=false, aspect_ratio=:equal))
