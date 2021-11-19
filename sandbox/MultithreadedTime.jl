#=
CPUTime is NOT thread safe
But that should come as no surprise as it's already been established here:
    - https://github.com/schmrlng/CPUTime.jl/issues/1
=#

using CPUTime

function runtime()
    t = rand()
    t0 = time()
    sleep(t)
    return (time()-t0, t)
end

function runCPUtime()
    t = rand()
    t0 = CPUtime_us()
    sleep(t)
    return ((CPUtime_us()-t0)*1e-6, t)
end

function TimeError(f::Function, N::Int)
    v = Vector{NTuple{2,Float64}}(undef, N)
    Threads.@threads for i in 1:N
        v[i] = f()
    end
    return mapreduce(t -> (t[2]-t[1])^2, +, v), v
end

BaseTimeError, v_base = TimeError(runtime, 10)

CPUTimeError, v_cpu = TimeError(runCPUtime, 10)

@show BaseTimeError # 0.000581
@show CPUTimeError # 3.12
