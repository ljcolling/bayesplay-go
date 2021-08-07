package bayesfactor

import (
	"math"

	. "pkg/distributions"
)

func CreateLikelihood(likelihood LikelihoodDefinition) Likelihood {

	var data Likelihood

	switch likelihood.Name {
	case "noncentral_d":
		d := likelihood.Params[0]
		n := likelihood.Params[1]
		fun := NoncentralDLikelihood(d, n)
		data.Function = fun
		data.Name = "noncentral_d"

	case "noncentral_d2":
		d := likelihood.Params[0]
		n1 := likelihood.Params[1]
		n2 := likelihood.Params[2]
		fun := NoncentralD2Likelihood(d, n1, n2)
		data.Function = fun
		data.Name = "noncentral_d2"

	case "normal":
		mean := likelihood.Params[0]
		sd := likelihood.Params[1]
		fun := NormalLikelihood(mean, sd)
		data.Function = fun
		data.Name = "normal"

	case "binomial":
		successes := likelihood.Params[0]
		trials := likelihood.Params[1]
		data.Function = BinomialLikelihood(successes, trials)
		data.Name = "binomial"

	case "noncentral_t":
		t := likelihood.Params[0]
		df := likelihood.Params[1]
		// fun := Noncentral_t_likelihood(t, df)
		data.Function = NoncentralTLikelihood(t, df)
		data.Name = "noncentral_t"

	case "student_t":
		mean := likelihood.Params[0]
		sd := likelihood.Params[1]
		df := likelihood.Params[2]
		fun := StudentTLikelihood(mean, sd, df)
		data.Function = fun
		data.Name = "student_t"
	}

	return data
}

func CreatePrior(priorDefinition PriorDefinition) Prior {
	var prior Prior

	switch priorDefinition.Name {
	case "cauchy":
		location := priorDefinition.Params[0]
		scale := priorDefinition.Params[1]
		min := priorDefinition.Params[2]
		max := priorDefinition.Params[3]
		prior = CauchyPrior(location, scale, min, max)

	case "normal":
		mean := priorDefinition.Params[0]
		sd := priorDefinition.Params[1]
		min := priorDefinition.Params[2]
		max := priorDefinition.Params[3]
		prior = NormalPrior(mean, sd, min, max)

	case "beta":
		alpha := priorDefinition.Params[0]
		beta := priorDefinition.Params[1]
		prior = BetaPrior(alpha, beta, 0, 1)

	case "uniform":
		alpha := priorDefinition.Params[0]
		beta := priorDefinition.Params[1]
		prior = UniformPrior(alpha, beta)

	case "student_t":
		mean := priorDefinition.Params[0]
		sd := priorDefinition.Params[1]
		df := priorDefinition.Params[2]
		min := priorDefinition.Params[3]
		max := priorDefinition.Params[4]
		prior = StudentTPrior(mean, sd, df, min, max)

	case "point":
		point := priorDefinition.Params[0]
		prior = PointPrior(point)
	}

	return prior
}

func Bayesfactor(likelihood LikelihoodDefinition, altprior PriorDefinition, nullprior PriorDefinition) (float64, error) {

	// data := CreateLikelihood(likelihood)
	// alt := CreatePrior(altprior)
	// null := CreatePrior(nullprior)

	altModel := Pp(likelihood, altprior)
	nullModel := Pp(likelihood, nullprior)

	bf := altModel.Auc / nullModel.Auc
	return bf, nil
}

// Types

type LikelihoodDefinition struct {
	Name   string
	Params []float64
}

type PriorDefinition struct {
	Name   string
	Params []float64
}

// Output types
// Predctive type

type Predictive struct {
	Function   func(x float64) float64
	Auc        float64
	Likelihood func(x float64) float64
	Prior      func(x float64) float64
}

// Prior type
type Prior struct {
	Function func(x float64) float64
	Name     string
	point    float64 // this is only used for the point prior because floating point :(
}

// Likelihood type
type Likelihood struct {
	Function func(x float64) float64
	Name     string
}

// Helper functions
func inrange(x float64, min float64, max float64) float64 {

	if x >= min && x <= max {
		return 1
	}

	return 0
}

func mult(likelihood func(x float64) float64, prior func(x float64) float64) func(x float64) float64 {
	return func(x float64) float64 {
		return likelihood(x) * prior(x)
	}
}

func Pp(likelihoodDef LikelihoodDefinition, priorDef PriorDefinition) Predictive {
	loops := 0
START:

	likelihood := CreateLikelihood(likelihoodDef)
	prior := CreatePrior(priorDef)
	var prod func(x float64) float64
	likelihoodFunction := likelihood.Function
	prod = mult(likelihoodFunction, prior.Function)
	var pred Predictive
	pred.Function = prod
	switch prior.Name {
	case "point":
		pred.Auc = likelihoodFunction(prior.point)
		goto END
	}
	switch likelihood.Name {
	case "binomial":
		pred.Auc = Integrate(prod, 0, 1)
	default:
		pred.Auc = Integrate(prod, math.Inf(-1), math.Inf(1))
	}

END:
	pred.Likelihood = likelihood.Function
	pred.Prior = prior.Function

	// noncentral t distributions have issues with extreme values
	// but this issue is restricted to observations of a particular sign
	// the easy fix for this is just to flip the observation
	if math.IsNaN(pred.Auc) {
		likelihoodDef.Params[0] = -likelihoodDef.Params[0]
		loops += 1
		if loops > 1 {
			// bail out to prevent infinite loop
			return pred
		}
		goto START
	}

	return pred

}

// normal likelihood

func NormalLikelihood(mean float64, sd float64) func(x float64) float64 {
	return func(x float64) float64 {
		return Dnorm(x, mean, sd)
	}
}

// student-t likelihood

func StudentTLikelihood(mean float64, sd float64, df float64) func(x float64) float64 {
	return func(x float64) float64 {
		return Scaled_shifted_t(x, mean, sd, df)
	}
}

// noncentral t likehood

func NoncentralTLikelihood(t float64, df float64) func(x float64) float64 {
	return func(x float64) float64 {
		return Dt(t, df, x)
	}
}

// noncentral d likelihood

func NoncentralDLikelihood(d float64, n float64) func(x float64) float64 {
	df := n - 1
	return func(x float64) float64 {
		return Dt(d*math.Sqrt(df+1), df, math.Sqrt(df+1)*x)
	}
}

func NoncentralD2Likelihood(d float64, n1 float64, n2 float64) func(x float64) float64 {
	return func(x float64) float64 {
		return Dt(d/math.Sqrt((1/n1)+(1/n2)), n1+n2-2, x*math.Sqrt((n1*n2)/(n1+n2)))
	}
}

// binomial likelihood

func BinomialLikelihood(successes float64, trials float64) func(x float64) float64 {
	return func(x float64) float64 {
		return Dbinom(successes, trials, x)
	}
}

// normal prior

func NormalPrior(mean float64, sd float64, min float64, max float64) Prior {

	// If max and max are +/-Inf then set K to 1
	// otherwise, integrate and normalize
	if min == math.Inf(-1) && max == math.Inf(1) {

		var prior Prior
		prior.point = 0
		prior.Function = func(x float64) float64 {
			return Dnorm(x, mean, sd)
		}
		prior.Name = "normal"
		return prior
	} else if (min == 0.0 && max == math.Inf(1)) || (min == math.Inf(-1) && max == 0.0) {
		k := 2.0
		var prior Prior
		prior.Function = func(x float64) float64 {
			return (Dnorm(x, mean, sd) * inrange(x, min, max)) * k
		}
		prior.Name = "normal"
		return prior
	} else {
		normal := func(x float64) float64 {
			return Dnorm(x, mean, sd) * inrange(x, min, max)
		}
		auc := Integrate(normal, math.Inf(-1), math.Inf(1))
		k := 1 / auc
		var prior Prior
		prior.Function = func(x float64) float64 {
			return (Dnorm(x, mean, sd) * inrange(x, min, max)) * k
		}
		prior.Name = "normal"
		return prior
	}
}

// student t prior

func StudentTPrior(mean float64, sd float64, df float64, min float64, max float64) Prior {

	// If max and max are +/-Inf then set K to 1
	// otherwise, integrate and normalize
	if min == math.Inf(-1) && max == math.Inf(1) {

		var prior Prior
		prior.point = 0
		prior.Function = func(x float64) float64 {
			return Scaled_shifted_t(x, mean, sd, df)
		}
		prior.Name = "student_t"
		return prior
	} else if (min == 0.0 && max == math.Inf(1)) || (min == math.Inf(-1) && max == 0.0) {
		k := 2.0
		var prior Prior
		prior.Function = func(x float64) float64 {
			return (Scaled_shifted_t(x, mean, sd, df) * inrange(x, min, max)) * k
		}
		prior.Name = "student_t"
		return prior
	} else {
		normal := func(x float64) float64 {
			return Scaled_shifted_t(x, mean, sd, df) * inrange(x, min, max)
		}
		auc := Integrate(normal, math.Inf(-1), math.Inf(1))
		k := 1 / auc
		var prior Prior
		prior.Function = func(x float64) float64 {
			return (Scaled_shifted_t(x, mean, sd, df) * inrange(x, min, max)) * k
		}
		prior.Name = "student_t"
		return prior
	}
}

// cauchy prior

func CauchyPrior(location float64, scale float64, min float64, max float64) Prior {

	// If max and max are +/-Inf then set K to 1
	// otherwise, integrate and normalize
	if min == math.Inf(-1) && max == math.Inf(1) {

		var prior Prior
		prior.point = 0
		prior.Function = func(x float64) float64 {
			return Dcauchy(x, location, scale)
		}
		prior.Name = "cauchy"
		return prior
	} else if (min == 0.0 && max == math.Inf(1)) || (min == math.Inf(-1) && max == 0.0) {
		k := 2.0
		var prior Prior
		prior.Function = func(x float64) float64 {
			return (Dcauchy(x, location, scale) * inrange(x, min, max)) * k
		}
		prior.Name = "cauchy"
		return prior
	} else {
		cauchy := func(x float64) float64 {
			return Dcauchy(x, location, scale) * inrange(x, min, max)
		}
		auc := Integrate(cauchy, math.Inf(-1), math.Inf(1))
		k := 1 / auc
		var prior Prior
		prior.Function = func(x float64) float64 {
			return (Dcauchy(x, location, scale) * inrange(x, min, max)) * k
		}
		prior.Name = "cauchy"
		return prior
	}
}

// beta prior

func BetaPrior(alpha float64, beta float64, min float64, max float64) Prior {

	var prior Prior
	prior.point = 0
	prior.Function = func(x float64) float64 {
		return Dbeta(x, alpha, beta) * inrange(x, min, max)
	}
	prior.Name = "beta"
	return prior
}

// point prior

func PointPrior(point float64) Prior {

	var prior Prior
	prior.Function = func(x float64) float64 {
		if x == point {
			return 1.0
		}
		return 0
	}
	prior.Name = "point"
	prior.point = point
	return prior
}

// uniform prior

func UniformPrior(alpha float64, beta float64) Prior {

	var prior Prior
	prior.point = 0
	prior.Function = func(x float64) float64 {
		return Dunif(x, alpha, beta)
	}
	prior.Name = "uniform"
	return prior
}
