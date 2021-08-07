package bayesfactor

import (
	"math"
	"testing"

	"github.com/google/go-cmp/cmp"
)

func compare(t *testing.T, got, want float64) {

	// const tolerance = .0001
	const tolerance = .001
	opt := cmp.Comparer(func(x, y float64) bool {
		diff := math.Abs(x - y)
		mean := math.Abs(x+y) / 2.0
		return (diff / mean) < tolerance
	})

	if !cmp.Equal(got, want, opt) {
		t.Fatalf("got %v, wanted %v", got, want)
	}
}

func TestBayesfactor(t *testing.T) {

	var likelihood LikelihoodDefinition // likelihood
	var altprior PriorDefinition        // alternative prior
	var nullprior PriorDefinition       // null prior
	//// likelihood
	likelihood.Name = "noncentral_d"
	likelihood.Params = []float64{2.03 / math.Sqrt(80), 80}
	//// alt prior
	altprior.Name = "cauchy"
	altprior.Params = []float64{0, 1, math.Inf(-1), math.Inf(1)}
	//// null prior
	nullprior.Name = "point"
	nullprior.Params = []float64{0.0}
	bf, _ := Bayesfactor(likelihood, altprior, nullprior)
	got := bf
	want := 1 / 1.557447
	compare(t, got, want)

	likelihood.Name = "noncentral_t"
	likelihood.Params = []float64{2.03, 79.0}
	//// alt prior
	altprior.Name = "cauchy"
	altprior.Params = []float64{0, 1 * math.Sqrt(80), math.Inf(-1), math.Inf(1)}
	//// null prior
	nullprior.Name = "point"
	nullprior.Params = []float64{0.0}
	bf, _ = Bayesfactor(likelihood, altprior, nullprior)
	got = bf
	want = 1 / 1.557447
	compare(t, got, want)

	likelihood.Name = "normal"
	likelihood.Params = []float64{5.5, 32.35}
	//// alt prior
	altprior.Name = "normal"
	altprior.Params = []float64{0, 13.3, 0, math.Inf(1)}
	//// null prior
	nullprior.Name = "point"
	nullprior.Params = []float64{0.0}
	bf, _ = Bayesfactor(likelihood, altprior, nullprior)
	got = bf
	want = 0.9745934
	compare(t, got, want)

	likelihood.Name = "normal"
	likelihood.Params = []float64{5, 10}
	//// alt prior
	altprior.Name = "uniform"
	altprior.Params = []float64{0, 20.0}
	//// null prior
	nullprior.Name = "point"
	nullprior.Params = []float64{0.0}
	bf, _ = Bayesfactor(likelihood, altprior, nullprior)
	got = bf
	want = 0.887226
	compare(t, got, want)

	likelihood.Name = "binomial"
	likelihood.Params = []float64{8, 11}
	//// alt prior
	altprior.Name = "beta"
	altprior.Params = []float64{2.5, 1}
	//// null prior
	nullprior.Name = "point"
	nullprior.Params = []float64{0.5}
	bf, _ = Bayesfactor(likelihood, altprior, nullprior)
	got = bf
	want = 1 / 0.6632996
	compare(t, got, want)

	likelihood.Name = "binomial"
	likelihood.Params = []float64{2, 10}
	//// alt prior
	altprior.Name = "normal"
	altprior.Params = []float64{0, 1, 0, 1}
	//// null prior
	nullprior.Name = "point"
	nullprior.Params = []float64{0.5}
	bf, _ = Bayesfactor(likelihood, altprior, nullprior)
	got = bf
	want = 2.327971
	compare(t, got, want)

	likelihood.Name = "student_t"
	likelihood.Params = []float64{5.47, 32.2, 119}
	//// alt prior
	altprior.Name = "student_t"
	altprior.Params = []float64{13.3, 4.93, 72, math.Inf(-1), math.Inf(1)}
	//// null prior
	nullprior.Name = "point"
	nullprior.Params = []float64{0}
	bf, _ = Bayesfactor(likelihood, altprior, nullprior)
	got = bf
	want = 0.9738
	compare(t, got, want)

	likelihood.Name = "noncentral_d2"
	likelihood.Params = []float64{-0.644110547740848, 15, 16}
	// alt prior
	altprior.Name = "cauchy"
	altprior.Params = []float64{0, 1, math.Inf(-1), math.Inf(1)}
	// null prior
	nullprior.Name = "point"
	nullprior.Params = []float64{0}
	bf, _ = Bayesfactor(likelihood, altprior, nullprior)
	got = bf
	want = 0.9709
	compare(t, got, want)

	// test with extreme values of for noncentral t distributions
	// bayesplay-go and the BayesFactor R package start to diverage
	// at effect sizes above about ~4.564355
	// this is because R uses an approximation of the noncentral t
	// for large values of t... this is currently not implemented
	// in bayesplay-go
	// 	likelihood.Name = "noncentral_d"
	// 	likelihood.Params = []float64{25.0 / math.Sqrt(30), 30}
	// 	altprior.Name = "cauchy"
	// 	altprior.Params = []float64{0, 1, math.Inf(-1), math.Inf(1)}
	//
	// 	nullprior.Name = "point"
	// 	nullprior.Params = []float64{0}
	// 	bf, _ = Bayesfactor(likelihood, altprior, nullprior)
	// 	got = bf
	// 	want = 1.37081e+18
	// 	compare(t, got, want)

}

// func BenchmarkPlots(b *testing.B) {
//
// 	mean := 0.0
// 	sd := 1.0
// 	min := mean - 4*sd // min range of plot
// 	max := mean + 4*sd // max range  of plot
// 	result := []interface{}{}
//
// 	likelihoodFunction := NormalLikelihood(mean, sd)
//
// 	step := (max - min) / 100
// 	x := min
// 	for i := 0; i < 101; i++ {
// 		y := likelihoodFunction(x)
// 		res := map[string]interface{}{"x": x, "y": y}
// 		result = append(result, res)
// 		x += step
// 	}
//
// }
//
// func BenchmarkDefault(b *testing.B) {
//
// 	var likelihood LikelihoodDefinition // likelihood
// 	var altprior PriorDefinition        // alternative prior
// 	var nullprior PriorDefinition       // null prior
// 	//// likelihood
// 	likelihood.Name = "noncentral_d"
// 	likelihood.Params = []float64{2.03 / math.Sqrt(80), 79.0}
// 	//// alt prior
// 	altprior.Name = "cauchy"
// 	altprior.Params = []float64{0, 1, math.Inf(-1), math.Inf(1)}
// 	//// null prior
// 	nullprior.Name = "point"
// 	nullprior.Params = []float64{0.0}
// 	Bayesfactor(likelihood, altprior, nullprior)
//
// }

// func BenchmarkPrediction(b *testing.B) {
//
// 	var likelihood LikelihoodDefinition // likelihood
// 	var altprior PriorDefinition        // alternative prior
// 	//// likelihood
// 	t := -10.0
// 	i := 0
// 	step := 0.1
// 	stop := 101
// 	var values []float64
// 	for i < stop {
// 		likelihood.Name = "noncentral_d"
// 		likelihood.Params = []float64{t / math.Sqrt(80), 79.0}
// 		//// alt prior
// 		altprior.Name = "cauchy"
// 		altprior.Params = []float64{0, 1, math.Inf(-1), math.Inf(1)}
//
// 		val := Pp(likelihood, altprior).Auc
// 		values = append(values, val)
// 		t += step
// 		i++
// 	}
// }

// func BenchmarkPrediction2(b *testing.B) {
//
// 	var likelihood LikelihoodDefinition // likelihood
// 	var altprior PriorDefinition        // alternative prior
// 	//// likelihood
// 	t := -2.0
// 	i := 0
// 	step := 0.1
// 	stop := 101
// 	// var values []float64
// 	values := [101]interface{}{}
// 	for i < stop {
// 		likelihood.Name = "noncentral_d"
// 		likelihood.Params = []float64{t / math.Sqrt(80), 79.0}
// 		//// alt prior
// 		altprior.Name = "cauchy"
// 		// altprior.Params = []float64{0, 1, math.Inf(-1), math.Inf(1)}
// 		altprior.Params = []float64{0, 1, -10, 10}
// 		val := Pp(likelihood, altprior).Auc
// 		res := map[string]interface{}{"x": t, "y": val}
// 		// values = append(values, res)
// 		values[i] = res
// 		t += step
// 		i++
// 	}
// }

