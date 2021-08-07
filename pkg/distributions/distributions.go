package distributions

import (
	"math"

	"gonum.org/v1/gonum/integrate/quad"
	"gonum.org/v1/gonum/stat/distuv"
	"scientificgo.org/special"
)

// consistent interface for statistica distributions

func findlimits(f func(x float64) float64) float64 {
	val := 0.0
	x := 0.0
	for !math.IsNaN(val) {
		val = f(x)
		x = x + 2
	}
	// val = 0
	for math.IsNaN(val) {
		val = f(x)
		x = x - .1
	}

	return x //- .2
}

// This is needed because of errors that sometimes result from
// integrating t distributions

func Integrate(f func(float64) float64, min float64, max float64) float64 {
	auc := quad.Fixed(f, min, max, 10000, nil, 100)
	return auc
}

func Dunif(x float64, min float64, max float64) float64 {
	dist := distuv.Uniform{
		Min: min,
		Max: max,
		Src: nil,
	}
	return dist.Prob(x)
}

func Dbinom(x float64, n float64, p float64) float64 {
	dist := distuv.Binomial{
		N:   n,
		P:   p,
		Src: nil,
	}
	return dist.Prob(x)
}

func Dbeta(x float64, shape1 float64, shape2 float64) float64 {
	dist := distuv.Beta{
		Alpha: shape1,
		Beta:  shape2,
		Src:   nil,
	}
	return dist.Prob(x)
}

func Scaled_shifted_t(x float64, mean float64, sd float64, df float64) float64 {
	dist := distuv.StudentsT{
		Mu:    mean,
		Sigma: sd,
		Nu:    df,
		Src:   nil,
	}
	return dist.Prob(x)
}

func Dnorm(x float64, mean float64, sd float64) float64 {
	dist := distuv.Normal{
		Mu:    mean,
		Sigma: sd,
		Src:   nil,
	}
	return dist.Prob(x)
}

func Dt(x float64, df float64, ncp float64) float64 {
	var x2 float64 = x * x
	var ncx2 float64 = ncp * ncp * x2
	var fac1 float64 = df + x2

	lgammadf1, _ := math.Lgamma(df + 1)
	lgamma_halfdf, _ := math.Lgamma(df / 2)

	trm1 := df/2.*math.Log(df) + lgammadf1
	trm1 -= df*math.Log(2) + ncp*ncp/2 + (df/2)*math.Log(fac1) + lgamma_halfdf

	var Px float64 = math.Exp(trm1)
	var valF float64 = ncx2 / (2 * fac1)

	trm1 = math.Sqrt(2) * ncp * x * special.HypPFQ([]float64{df/2 + 1}, []float64{1.5}, valF)
	trm1 /= fac1 * math.Gamma((df+1)/2)

	var trm2 float64 = special.HypPFQ([]float64{(df + 1) / 2}, []float64{0.5}, valF)
	trm2 /= math.Sqrt(fac1) * math.Gamma(df/2+1)
	Px *= trm1 + trm2
  if math.IsNaN(Px) {
    return 0
  }
	return Px
}

func Dcauchy(x float64, location float64, scale float64) float64 {
	dist := distuv.StudentsT{
		Mu:    location,
		Sigma: scale,
		Nu:    1,
		Src:   nil,
	}
	return dist.Prob(x)
}
