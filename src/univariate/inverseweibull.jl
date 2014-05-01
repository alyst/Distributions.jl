# Inverse Weibull Distribution
# for information see e.g. M.-S. Khan, Theoretical Analysis of Inverse Weibull Distribution, WSEAS Transactions on Mathematics 2008

immutable InverseWeibull <: ContinuousUnivariateDistribution
    shape::Float64
    scale::Float64
    function InverseWeibull(sh::Real, sc::Real)
    	zero(sh) < sh && zero(sc) < sc || error("Both shape and scale must be positive")
    	new(float64(sh), float64(sc))
    end
end

InverseWeibull(sh::Real) = InverseWeibull(sh, 1.0)

## Support
@continuous_distr_support InverseWeibull 0.0 Inf

## Properties
mean(d::InverseWeibull) = d.shape > 1.0 ? d.scale * gamma(1.0 - 1.0 / d.shape) : Inf
median(d::InverseWeibull) = d.scale * log(2)^(-1.0 / d.shape)

mode(d::InverseWeibull) = (ik = -1.0/d.shape; d.scale * (1.0-ik)^ik)

var(d::InverseWeibull) = d.shape > 2.0 ? d.scale^2 * gamma(1.0 - 2.0 / d.shape) - mean(d)^2 : NaN

function skewness(d::InverseWeibull)
    d.shape <= 3.0 && return(NaN)
    tmp_mean = mean(d)
    tmp_var = var(d)
    tmp = gamma(1.0 - 3.0 / d.shape) * d.scale^3
    tmp -= 3.0 * tmp_mean * tmp_var
    tmp -= tmp_mean^3
    return tmp / tmp_var / sqrt(tmp_var)
end

function kurtosis(d::InverseWeibull)
    d.shape <= 4.0 && return(NaN)
    λ, k = d.scale, d.shape
    μ = mean(d)
    σ = std(d)
    γ = skewness(d)
    den = λ^4 * gamma(1.0 - 4.0 / k) -
          4.0 * γ * σ^3 * μ -
          6.0 * μ^2 * σ^2 - μ^4
    num = σ^4
    return den / num - 3.0
end

function entropy(d::InverseWeibull)
    λ, k = d.scale, d.shape
    return (k + 1.0) * (log(λ) - digamma(1.0)/k) - log(λ * k) + 1.0
end

## Functions
function pdf(d::InverseWeibull, x::Real)
    x < zero(x) && return(0.0)
    a = d.scale/x
    d.shape/d.scale * a^(d.shape+1.0) * exp(-a^d.shape)
end
function logpdf(d::InverseWeibull, x::Real)
    x < zero(x) && return(-Inf)
    a = d.scale/x
    log(d.shape/d.scale) + (d.shape+1.0)*log(a) - a^d.shape
end

cdf(d::InverseWeibull, x::Real) = x <= zero(x) ? 0.0 : exp(-((d.scale / x)^d.shape))
ccdf(d::InverseWeibull, x::Real) = x <= zero(x) ? 1.0 : -expm1(-((d.scale / x)^d.shape))
logcdf(d::InverseWeibull, x::Real) = x <= zero(x) ? -Inf : -(d.scale / x)^d.shape
logccdf(d::InverseWeibull, x::Real) = x <= zero(x) ? 0.0 :  log1mexp(-((d.scale / x)^d.shape))

quantile(d::InverseWeibull, p::Real) = @checkquantile p d.scale*(-log(p))^(-1/d.shape)
cquantile(d::InverseWeibull, p::Real) = @checkquantile p d.scale*(-log1p(-p))^(-1/d.shape)
invlogcdf(d::InverseWeibull, lp::Real) = lp > zero(lp) ? NaN : d.scale*(-lp)^(-1/d.shape)
invlogccdf(d::InverseWeibull, lp::Real) = lp > zero(lp) ? NaN : d.scale*(-log1mexp(lp))^(-1/d.shape)


function gradloglik(d::InverseWeibull, x::Float64)
  insupport(InverseWeibull, x) ? -(d.shape + 1.0) / x + d.shape * (d.scale^d.shape) * x^(-d.shape - 1.0)  : 0.0
end

## Sampling
rand(d::InverseWeibull) = d.scale*Base.Random.randmtzig_exprnd()^(-1/d.shape)
