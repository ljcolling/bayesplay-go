package main

import (
	"errors"
	"fmt"

	"pkg/bayesfactor"
	"pkg/distributions"

	"math"
	"sort"
	"syscall/js"
)

//// alias imported function names
var print = fmt.Println

// main distributions
var dt = distributions.Dt
var dnorm = distributions.Dnorm
var dunif = distributions.Dunif
var dbinom = distributions.Dbinom
var dbeta = distributions.Dbeta
var scaledShiftedT = distributions.Scaled_shifted_t
var dcauchy = distributions.Dcauchy

var integrate = distributions.Integrate
var sqrt = math.Sqrt
var bf = bayesfactor.Bayesfactor

// MinMax finds the minimum and the maxium of array
func MinMax(array []float64) (float64, float64) {
	var max float64 = array[0]
	var min float64 = array[0]
	for _, value := range array {
		if max < value {
			max = value
		}
		if min > value {
			min = value
		}
	}
	return min, max
}

func main() {

	fmt.Println("Loaded WASM...")

	js.Global().Set("bayesfactor", js.FuncOf(bfWrapper))
	js.Global().Set("dnorm", js.FuncOf(dnormWrapper))
	js.Global().Set("dbeta", js.FuncOf(dbetaWrapper))
	js.Global().Set("dbinom", js.FuncOf(dbinomWrapper))
	js.Global().Set("scaled_shifted_t", js.FuncOf(scaledShiftedTWrapper))
	js.Global().Set("dnormPlot", js.FuncOf(dnormPlotWrapper))
	js.Global().Set("dnormPlotPrior", js.FuncOf(dnormPriorPlotWrapper))
	js.Global().Set("student_tPlotPrior", js.FuncOf(studentTPriorPlotWrapper))
	js.Global().Set("dbinomPlot", js.FuncOf(dbinomPlotWrapper))
	js.Global().Set("dbetaPlotPrior", js.FuncOf(dbetaPriorPlotWrapper))
	js.Global().Set("scaled_shifted_tPlot", js.FuncOf(scaledShiftedTPlotWrapper))
	js.Global().Set("noncentral_dPlot", js.FuncOf(noncentralDPlotWrapper))
	js.Global().Set("noncentral_d2Plot", js.FuncOf(noncentralD2PlotWrapper))
	js.Global().Set("noncentral_tPlot", js.FuncOf(noncentralTPlotWrapper))
	js.Global().Set("cauchyPlot_Prior", js.FuncOf(cauchyPriorPlotWrapper))
	js.Global().Set("uniformPriorPlot", js.FuncOf(uniformPriorPlotWrapper))
	js.Global().Set("computeAll", js.FuncOf(computeWrapper))

	js.Global().Set("loaded", "true")
	<-make(chan bool)

}

func getParam(arg js.Value, field string) (js.Value, error) {

	if arg.Get(field).IsUndefined() {
		return arg, errors.New("missing field")
	} else if arg.Get(field).IsNull() {
		return arg, errors.New("missing field")
	} else if arg.Get(field).IsNaN() {
		return arg, errors.New("missing field")
	} else {
		return arg.Get(field), nil
	}
}

func getMinMax(arg js.Value, field string, likelihoodName string) float64 {

	var val float64
	var defaultval float64

	if likelihoodName == "binomial" && field == "min" {
		defaultval = 0
	} else if likelihoodName == "binomial" && field == "max" {
		defaultval = 1
	} else if likelihoodName != "binomial" && field == "min" {
		defaultval = math.Inf(-1)
	} else if likelihoodName != "binomial" && field == "max" {
		defaultval = math.Inf(1)
	}

	if arg.Get("parameters").Get(field).IsNull() {
		val = defaultval
	} else if arg.Get("parameters").Get(field).IsNaN() {
		val = defaultval
	} else if arg.Get("parameters").Get(field).IsUndefined() {
		val = defaultval
	} else {
		val = arg.Get("parameters").Get(field).Float()
	}
	return val

}

func seq(min float64, max float64) [101]float64 {
	step := (max - min) / (100)
	var values [101]float64
	t := min
	for i := 0; i < 101; i++ {
		values[i] = t
		t += step
	}
	values[100] = max

	return values
}

func seqShort(min float64, max float64) []float64 {
	step := (max - min) / (100)
	var values []float64
	t := min
	for i := min; i < max; i += step {
		values = append(values, t)
		t += step
	}
	values = append(values, max)

	return values
}

func seqShort20(min, max float64) [20]float64 {
	steps := 20
	step := (max - min) / float64(steps)
	var values [20]float64
	t := min
	for i := 0; i < 20; i++ {
		values[i] = t
		t += step
	}

	values[19] = max

	return values

}

func seqSteps(min float64, max float64, stepSize float64) []float64 {

	var values []float64
	i := 0
	for v := min; v < max; v += stepSize {
		values = append(values, v)
		i++
	}

	return values
}

func generatePredictions_old(likelihood bayesfactor.LikelihoodDefinition, prior bayesfactor.PriorDefinition, minvalue float64, maxvalue float64, ob float64) interface{} {
	// var predicts []float64
	result := []interface{}{}

	var newLikelihood bayesfactor.LikelihoodDefinition
	newLikelihood.Name = likelihood.Name
	newLikelihood.Params = likelihood.Params
	// return if it's a binomial likelihood
	if likelihood.Name == "binomial" {
		trials := newLikelihood.Params[1]
		observations := seqSteps(0, trials+1, 1)
		// Prediction := bayesfactor.Pp(bayesfactor.CreateLikelihood(likelihood), bayesfactor.CreatePrior(prior))
		// res := map[string]interface{}{"x": likelihood.Params[0], "y": Prediction.Auc}
		// result = append(result, res)
		for _, ob := range observations {
			newLikelihood.Params[0] = ob
			Prediction := bayesfactor.Pp(newLikelihood, prior)
			res := map[string]interface{}{"x": ob, "y": Prediction.Auc}
			// res := map[string]interface{}{"x": ob, "y": 0}
			// predicts = append(predicts, Prediction.Auc)
			result = append(result, res)
		}

		return result

	}

	// observations := seq_short(minvalue, maxvalue)
	observations := seqShort(minvalue, maxvalue)
	observations = append(observations, ob)
	// fmt.Println(observations)
	sort.Float64s(observations)

	for _, ob := range observations {
		newLikelihood.Params[0] = ob
		Prediction := bayesfactor.Pp(newLikelihood, prior)
		res := map[string]interface{}{"x": ob, "y": Prediction.Auc}
		// predicts = append(predicts, Prediction.Auc)
		result = append(result, res)
	}
	return result

}

func generatePredictions(
	likelihood bayesfactor.LikelihoodDefinition,
	altprior bayesfactor.PriorDefinition,
	nullprior bayesfactor.PriorDefinition,
	minvalue float64,
	maxvalue float64,
	currentObservation float64,
	bf float64,
) (interface{}, interface{}) {

	// the minvalue and maxvalue need to be calculated here
	// the currentObservation also beeds to be calcualted here
	// the current bf also needs to be calculated
	// because the noncentral t distribution doesn't like very large
	// positive t values, the marginal likelihoods are caluculated from
	// the negative t values
	// if it's a noncentral d distribution then limit the range to about +/- 5

	if likelihood.Name == "noncentral_d" {
		minvalue, maxvalue = MinMax([]float64{-3, 3, currentObservation + 1.0, -currentObservation + 1.0})
	}

	comparison := []interface{}{}
	ratio := []interface{}{}

	var newLikelihood bayesfactor.LikelihoodDefinition
	newLikelihood.Name = likelihood.Name
	newLikelihood.Params = likelihood.Params

	var observations []float64
	if likelihood.Name == "binomial" {
		trials := newLikelihood.Params[1]
		observations = seqSteps(0, trials+1, 1)
	} else {
		observations = seqShort(minvalue, maxvalue)
		observations = append(observations, currentObservation)
		sort.Float64s(observations)
	}

	for _, ob := range observations {
		newLikelihood.Params[0] = ob
		altPrediction := bayesfactor.Pp(newLikelihood, altprior).Auc
		nullPrediction := bayesfactor.Pp(newLikelihood, nullprior).Auc
		// if math.IsNaN(altPrediction) || math.IsNaN(nullPrediction) {
		// 	newLikelihood.Params[0] = -ob
		// 	altPrediction = bayesfactor.Pp(bayesfactor.CreateLikelihood(newLikelihood), bayesfactor.CreatePrior(altprior)).Auc
		// 	nullPrediction = bayesfactor.Pp(bayesfactor.CreateLikelihood(newLikelihood), bayesfactor.CreatePrior(nullprior)).Auc
		// }
		thisBf := math.Log10(altPrediction) - math.Log10(nullPrediction)
		if math.Abs(math.Log10(bf)) < 50 {
			ratioValue := map[string]interface{}{"x": ob, "y": thisBf}
			ratio = append(ratio, ratioValue)
			altValue := map[string]interface{}{"x": ob, "y": altPrediction, "type": "Alternative model"}
			comparison = append(comparison, altValue)
			nullValue := map[string]interface{}{"x": ob, "y": nullPrediction, "type": "Null model"}
			comparison = append(comparison, nullValue)
		}
	}

	// result := map[string]interface{}{
	// 	"comparison": comparison,
	// 	"ratio":      ratio}

	return comparison, ratio
}

func dnormWrapper(this js.Value, args []js.Value) interface{} {
	x := args[0].Float()
	mean := args[1].Float()
	sd := args[2].Float()
	return dnorm(x, mean, sd)
}

var normalLikelihood = bayesfactor.NormalLikelihood

// func compute_wrapper(this js.Value, args []js.Value) {
func computeWrapper(this js.Value, args []js.Value) interface{} {

	fullmodel := make([]js.Value, 1)
	fullmodel[0] = args[0]
	likelihood, _ := parseLikelihood(fullmodel)
	altprior, _ := parsePrior(args, "altpriorDef", likelihood.Name)
	nullprior, _ := parsePrior(fullmodel, "nullpriorDef", likelihood.Name)

	// compute the bf
	bf := bfWrapper(this, fullmodel)
	fmt.Println(likelihood.Name)

	// get the likelihood plot data
	var likelihoodPlotData interface{}
	switch likelihood.Name {
	case "normal":
		likelihoodPlotData = dnormPlot(likelihood.Params[0], likelihood.Params[1])
	case "student_t":
		likelihoodPlotData = scaledShiftedTPlot(likelihood.Params[0], likelihood.Params[1], likelihood.Params[2])
	case "binomial":
		likelihoodPlotData = dbinomPlot(likelihood.Params[0], likelihood.Params[1])
	case "noncentral_t":
		likelihoodPlotData = noncentralTPlot(likelihood.Params[0], likelihood.Params[1])
	case "noncentral_d":
		likelihoodPlotData = noncentralDPlot(likelihood.Params[0], likelihood.Params[1])
	case "noncentral_d2":
		likelihoodPlotData = noncentralD2Plot(likelihood.Params[0], likelihood.Params[1], likelihood.Params[2])
	}

	observation := likelihood.Params[0]

	var altpriorPlotData interface{}
	switch altprior.Name {
	case "normal":
		altpriorPlotData = dnormPlotPrior(altprior.Params[0], altprior.Params[1], altprior.Params[2], altprior.Params[3])
	case "student_t":
		altpriorPlotData = studentTPriorPlot(altprior.Params[0], altprior.Params[1], altprior.Params[2], altprior.Params[3], altprior.Params[4])
	case "beta":
		altpriorPlotData = dbetaPlotPrior(altprior.Params[0], altprior.Params[1])
	case "cauchy":
		altpriorPlotData = cauchyPriorPlot(altprior.Params[0], altprior.Params[1], altprior.Params[2], altprior.Params[3])
	case "uniform":
		altpriorPlotData = uniformPriorPlot(altprior.Params[0], altprior.Params[1])

	}

	var nullpriorPlotData interface{}
	switch nullprior.Name {
	case "normal":
		nullpriorPlotData = dnormPlotPrior(nullprior.Params[0], nullprior.Params[1], nullprior.Params[2], nullprior.Params[3])
	case "student_t":
		nullpriorPlotData = studentTPriorPlot(nullprior.Params[0], nullprior.Params[1], nullprior.Params[2], nullprior.Params[3], nullprior.Params[4])
	case "beta":
		nullpriorPlotData = dbetaPlotPrior(nullprior.Params[0], nullprior.Params[1])
	case "cauchy":
		nullpriorPlotData = cauchyPriorPlot(nullprior.Params[0], nullprior.Params[1], nullprior.Params[2], nullprior.Params[3])
	case "uniform":
		nullpriorPlotData = uniformPriorPlot(nullprior.Params[0], nullprior.Params[1])
	case "point":
		result := []interface{}{}
		res := map[string]interface{}{"x": nullprior.Params[0], "y": 1}
		result = append(result, res)
		nullpriorPlotData = result

	}

	// TODO: Things that still need adding
	// [ ] Visual comparison
	// [ ] Posterior plots
	// [ ] Anything else

	// this is for the visual comparison!
	// first decide on the limits of the prediction interval
	// I'll just do a quick one for binomial

	// fmt.Println("Find the minimum and max values")
	// find the limits of the likelihood function
	likelihoodLimitsXmin := likelihoodPlotData.([]interface{})[0].(map[string]interface{})["x"].(float64)
	likelihoodLimitsXmax := likelihoodPlotData.([]interface{})[100].(map[string]interface{})["x"].(float64)
	fmt.Println(likelihoodLimitsXmin)
	fmt.Println(likelihoodLimitsXmax)

	// find the limits of the alt prior
	altpriorLimitsXmin := altpriorPlotData.([]interface{})[0].(map[string]interface{})["x"].(float64)
	altpriorLimitsXmax := altpriorPlotData.([]interface{})[100].(map[string]interface{})["x"].(float64)
	fmt.Println(altpriorLimitsXmin)
	fmt.Println(altpriorLimitsXmax)

	var nullpriorLimitsXmin float64
	var nullpriorLimitsXmax float64
	if nullprior.Name == "point" {
		nullpriorLimitsXmin = altpriorLimitsXmin
		nullpriorLimitsXmax = altpriorLimitsXmax
	} else {
		nullpriorLimitsXmin = nullpriorPlotData.([]interface{})[0].(map[string]interface{})["x"].(float64)
		nullpriorLimitsXmax = nullpriorPlotData.([]interface{})[100].(map[string]interface{})["x"].(float64)
	}
	fmt.Println(nullpriorLimitsXmin)
	fmt.Println(nullpriorLimitsXmax)

	xmin, _ := MinMax([]float64{likelihoodLimitsXmin, altpriorLimitsXmin, nullpriorLimitsXmin})
	_, xmax := MinMax([]float64{likelihoodLimitsXmax, altpriorLimitsXmax, nullpriorLimitsXmax})

	_, lim := MinMax([]float64{math.Abs(xmin), math.Abs(xmax)})

	if xmin < 0 {
		xmin = lim * -1
	} else {
		xmin = lim * 1
	}

	if xmax < 0 {
		xmax = lim * -1
	} else {
		xmax = lim * 1
	}

	altPriorProd := bayesfactor.Pp(likelihood, altprior)
	nullPriorProd := bayesfactor.Pp(likelihood, nullprior)
	altPoint := altPriorProd.Auc
	nullPoint := nullPriorProd.Auc

	altPosteriorPlot := []interface{}{}
	altPosteriorValues := seq(altpriorLimitsXmin, altpriorLimitsXmax)
	for _, x := range altPosteriorValues {
		res := map[string]interface{}{"x": x, "y": altPriorProd.Function(x) / altPriorProd.Auc}
		altPosteriorPlot = append(altPosteriorPlot, res)
	}

	nullPosteriorPlot := []interface{}{}
	nullPosteriorValues := seq(nullpriorLimitsXmin, nullpriorLimitsXmax)
	for _, x := range nullPosteriorValues {
		res := map[string]interface{}{"x": x, "y": nullPriorProd.Function(x) / nullPriorProd.Auc}
		nullPosteriorPlot = append(nullPosteriorPlot, res)
	}
	// observation := likelihood.Params[0]

	comparison, ratio := generatePredictions(
		likelihood,
		altprior,
		nullprior,
		xmin,
		xmax,
		observation,
		altPoint/nullPoint)

	// fmt.Println(predictions)
	// res := map[string]interface{}{"x": x, "y": y}
	// result := []interface{}{}
	// result := map[string]interface{}{
	result := map[string]interface{}{
		"bf":                    bf,
		"likelihoodPlotData":    likelihoodPlotData,
		"altpriorPlotData":      altpriorPlotData,
		"nullpriorPlotData":     nullpriorPlotData,
		"altpriorLims":          map[string]interface{}{"xmin": altpriorLimitsXmin, "xmax": altpriorLimitsXmax},
		"xmin":                  altpriorLimitsXmin,
		"xmax":                  altpriorLimitsXmax,
		"names":                 map[string]interface{}{"likelihoodName": likelihood.Name, "alt": altprior.Name, "null": nullprior.Name},
		"observation":           observation,
		"altpoint":              altPoint,
		"nullpoint":             nullPoint,
		"comparison":            comparison,
		"ratio":                 ratio,
		"altposteriorPlotData":  altPosteriorPlot,
		"nullposteriorPlotData": nullPosteriorPlot,
	}
	// result = append(
	// 	result,
	// 	bf,
	// 	likelihoodPlotData,
	// 	altpriorPlotData,
	// 	nullpriorPlotData,
	// 	likelihood.Name,
	// 	altprior.Name,
	// 	nullprior.Name,
	// 	altPredictions,
	// 	nullPredictions,
	// )
	return result
}

func dnormPlotWrapper(this js.Value, args []js.Value) interface{} {
	mean := args[0].Float()
	sd := args[1].Float()
	return dnormPlot(mean, sd)
}

func dnormPlot(mean float64, sd float64) interface{} {

	min := mean - 4*sd // min range of plot
	max := mean + 4*sd // max range  of plot
	result := []interface{}{}

	likelihoodFunction := normalLikelihood(mean, sd)

	step := (max - min) / 100
	x := min
	for i := 0; i < 101; i++ {
		y := likelihoodFunction(x)
		res := map[string]interface{}{"x": x, "y": y}
		result = append(result, res)
		x += step
	}

	return result

}

var normalPrior = bayesfactor.NormalPrior

func dnormPriorPlotWrapper(this js.Value, args []js.Value) interface{} {

	mean := args[0].Float()
	sd := args[1].Float()
	var min float64
	var max float64
	if args[2].IsNull() {
		min = math.Inf(-1)
	} else {
		min = args[2].Float()
	}
	if args[3].IsNull() {
		max = math.Inf(1)
	} else {
		max = args[3].Float()
	}
	return dnormPlotPrior(mean, sd, min, max)
}

func dnormPlotPrior(mean float64, sd float64, min float64, max float64) interface{} {

	minRange := mean - 4*sd // min range of plot
	maxRange := mean + 4*sd // max range  of plot
	result := []interface{}{}

	priorFunction := bayesfactor.NormalPrior(mean, sd, min, max).Function

	step := (maxRange - minRange) / 100
	x := minRange
	for i := 0; i < 101; i++ {
		y := priorFunction(x)
		res := map[string]interface{}{"x": x, "y": y}
		result = append(result, res)
		x += step
	}

	return result

}

func studentTPriorPlotWrapper(this js.Value, args []js.Value) interface{} {

	mean := args[0].Float()
	sd := args[1].Float()
	df := args[2].Float()
	var min float64
	var max float64
	if args[3].IsNull() {
		min = math.Inf(-1)
	} else {
		min = args[3].Float()
	}
	if args[4].IsNull() {
		max = math.Inf(1)
	} else {
		max = args[4].Float()
	}

	return studentTPriorPlot(mean, sd, df, min, max)
}

func studentTPriorPlot(mean float64, sd float64, df float64, min float64, max float64) interface{} {

	minRange := mean - 4*sd // min range of plot
	maxRange := mean + 4*sd // max range  of plot
	result := []interface{}{}

	priorFunction := bayesfactor.StudentTPrior(mean, sd, df, min, max).Function

	step := (maxRange - minRange) / 100
	x := minRange
	for i := 0; i < 101; i++ {
		y := priorFunction(x)
		res := map[string]interface{}{"x": x, "y": y}
		result = append(result, res)
		x += step
	}

	return result

}

func cauchyPriorPlotWrapper(this js.Value, args []js.Value) interface{} {

	location := args[0].Float()
	scale := args[1].Float()

	var min float64
	var max float64
	if args[2].IsNull() {
		min = math.Inf(-1)
	} else {
		min = args[2].Float()
	}
	if args[3].IsNull() {
		max = math.Inf(1)
	} else {
		max = args[3].Float()
	}
	return cauchyPriorPlot(location, scale, min, max)
}

func cauchyPriorPlot(location float64, scale float64, min float64, max float64) interface{} {

	result := []interface{}{}

	minRange := location - 4*scale // min range of plot
	maxRange := location + 4*scale // max range  of plot

	priorFunction := bayesfactor.CauchyPrior(location, scale, min, max).Function

	step := (maxRange - minRange) / 100
	x := minRange
	for i := 0; i < 101; i++ {
		y := priorFunction(x)
		res := map[string]interface{}{"x": x, "y": y}
		result = append(result, res)
		x += step
	}

	return result
}

var binomialLikelihood = bayesfactor.BinomialLikelihood

func dbinomPlotWrapper(this js.Value, args []js.Value) interface{} {
	x := args[0].Float()
	n := args[1].Float()
	return dbinomPlot(x, n)
}

func dbinomPlot(x float64, n float64) interface{} {

	min := 0.0
	max := 1.0
	result := []interface{}{}

	likelihoodFunction := binomialLikelihood(x, n)

	step := (max - min) / 100
	p := min
	for i := 0; i < 101; i++ {
		y := likelihoodFunction(p)

		if math.IsNaN(y) {
			if i == 0 {
				y = likelihoodFunction(p + step)
			}
			if i == 100 {
				y = likelihoodFunction(p - step)
				p = 1
			}
		}

		res := map[string]interface{}{"x": p, "y": y}
		result = append(result, res)
		p += step
	}

	return result
}

func dbetaPriorPlotWrapper(this js.Value, args []js.Value) interface{} {

	alpha := args[0].Float()
	beta := args[1].Float()
	return dbetaPlotPrior(alpha, beta)
}

func dbetaPlotPrior(alpha float64, beta float64) interface{} {

	min := 0.0
	max := 1.0
	result := []interface{}{}

	priorFunction := bayesfactor.BetaPrior(alpha, beta, min, max).Function

	step := (max - min) / 100
	x := min
	for i := 0; i < 101; i++ {
		y := priorFunction(x)
		if math.IsNaN(y) {
			if i == 0 {
				y = priorFunction(x + step)
			}
			if i == 100 {
				y = priorFunction(x - step)
				x = 1
			}
		}

		res := map[string]interface{}{"x": x, "y": y}
		result = append(result, res)
		x += step
	}
	val := result[99]
	fmt.Println(val)

	return result
}

func inrange(x float64, min float64, max float64) float64 {
	if x >= min && x <= max {
		return 1
	}
	return 0
}

func uniformPriorPlotWrapper(this js.Value, args []js.Value) interface{} {

	alpha := args[0].Float()
	beta := args[1].Float()
	return uniformPriorPlot(alpha, beta)
}

func uniformPriorPlot(alpha float64, beta float64) interface{} {

	diff := math.Abs(alpha - beta)
	if diff <= 1 {
		diff = 2
	}
	min := alpha - (1.05 * diff)
	max := beta + (1.05 * diff)
	result := []interface{}{}

	priorFunction := bayesfactor.UniformPrior(alpha, beta).Function
	step := (max - min) / 100
	x := min
	for i := 0; i < 101; i++ {
		y := priorFunction(x)
		// y := inrange(x, alpha, beta)
		res := map[string]interface{}{"x": x, "y": y}
		result = append(result, res)
		x += step
	}

	return result
}

var studentTLikelihood = bayesfactor.StudentTLikelihood

func scaledShiftedTPlotWrapper(this js.Value, args []js.Value) interface{} {

	mean := args[0].Float()
	sd := args[1].Float()
	df := args[2].Float()
	return scaledShiftedTPlot(mean, sd, df)
}

func scaledShiftedTPlot(mean float64, sd float64, df float64) interface{} {

	min := mean - 4*sd // min range of plot
	max := mean + 4*sd // max range  of plot
	result := []interface{}{}

	likelihoodFunction := studentTLikelihood(mean, sd, df)

	step := (max - min) / 100
	x := min
	for i := 0; i < 101; i++ {
		y := likelihoodFunction(x)
		res := map[string]interface{}{"x": x, "y": y}
		result = append(result, res)
		x += step
	}

	return result

}

var noncentralDLikelihood = bayesfactor.NoncentralDLikelihood

func noncentralDPlotWrapper(this js.Value, args []js.Value) interface{} {

	d := args[0].Float()
	n := args[1].Float()
	return noncentralDPlot(d, n)

}

func noncentralDPlot(d float64, n float64) interface{} {
	df := n - 1
	variance := (df+df+2)/((df+1)*(df+1)) + ((d * d) / (2 * (df + df + 2)))
	sd := sqrt(variance)
	min := d - 4*sd // min range of plot
	max := d + 4*sd // max range  of plot
	result := []interface{}{}

	likelihoodFunction := noncentralDLikelihood(d, n)

	step := (max - min) / 100
	x := min
	for i := 0; i < 101; i++ {
		y := likelihoodFunction(x)
		res := map[string]interface{}{"x": x, "y": y}
		result = append(result, res)
		x += step
	}

	return result
}

var noncentralD2Likelihood = bayesfactor.NoncentralD2Likelihood

func noncentralD2PlotWrapper(this js.Value, args []js.Value) interface{} {

	d := args[0].Float()
	n1 := args[1].Float()
	n2 := args[2].Float()
	return noncentralD2Plot(d, n1, n2)

}

func noncentralD2Plot(d float64, n1 float64, n2 float64) interface{} {
	variance := (n1+n2)/((n1)*(n2)) + ((d * d) / (2 * (n1 + n2)))
	sd := sqrt(variance)
	min := d - 4*sd // min range of plot
	max := d + 4*sd // max range  of plot
	result := []interface{}{}

	likelihoodFunction := noncentralD2Likelihood(d, n1, n2)

	step := (max - min) / 100
	x := min
	for i := 0; i < 101; i++ {
		y := likelihoodFunction(x)
		res := map[string]interface{}{"x": x, "y": y}
		result = append(result, res)
		x += step
	}

	return result
}

var noncentralTLikelihood = bayesfactor.NoncentralTLikelihood

func noncentralTPlotWrapper(this js.Value, args []js.Value) interface{} {

	t := args[0].Float()
	df := args[1].Float()
	return noncentralTPlot(t, df)
}

func noncentralTPlot(t float64, df float64) interface{} {

	d := t * sqrt(df+1)
	variance := (df+df+2)/((df+1)*(df+1)) + ((d * d) / (2 * (df + df + 2)))
	sd := sqrt(variance)
	min := t - 4*sd // min range of plot
	max := t + 4*sd // max range  of plot
	result := []interface{}{}

	likelihoodFunction := noncentralTLikelihood(t, df)

	step := (max - min) / 100
	x := min
	for i := 0; i < 101; i++ {
		y := likelihoodFunction(x)
		res := map[string]interface{}{"x": x, "y": y}
		result = append(result, res)
		x += step
	}

	return result
}

func dbetaWrapper(this js.Value, args []js.Value) interface{} {
	x := args[0].Float()
	shape1 := args[1].Float()
	shape2 := args[2].Float()
	return dbeta(x, shape1, shape2)
}

func dbinomWrapper(this js.Value, args []js.Value) interface{} {
	x := args[0].Float()
	n := args[1].Float()
	p := args[2].Float()
	return dbinom(x, n, p)
}

func scaledShiftedTWrapper(this js.Value, args []js.Value) interface{} {
	if len(args) < 4 {
		return nil
	}
	var x float64
	var mean float64
	var sd float64
	var df float64
	x = args[0].Float()
	mean = args[1].Float()
	sd = args[2].Float()
	df = args[3].Float()
	return scaledShiftedT(x, mean, sd, df)
}

func parseLikelihood(args []js.Value) (bayesfactor.LikelihoodDefinition, error) {

	var likelihood bayesfactor.LikelihoodDefinition // likelihood

	likelihoodObj, err := getParam(args[0], "likelihoodDef")

	likelihoodName := likelihoodObj.Get("distribution").String()

	switch likelihoodName {
	case "normal":
		likelihood.Name = "normal"
		mean := likelihoodObj.Get("parameters").Get("mean").Float()
		sd := likelihoodObj.Get("parameters").Get("sd").Float()
		likelihood.Params = []float64{mean, sd}
	case "student t":
		likelihood.Name = "student_t"
		mean := likelihoodObj.Get("parameters").Get("mean").Float()
		sd := likelihoodObj.Get("parameters").Get("sd").Float()
		df := likelihoodObj.Get("parameters").Get("df").Float()
		likelihood.Params = []float64{mean, sd, df}
	case "binomial":
		likelihood.Name = "binomial"
		successes := likelihoodObj.Get("parameters").Get("successes").Float()
		trials := likelihoodObj.Get("parameters").Get("trials").Float()
		likelihood.Params = []float64{successes, trials}
	case "noncentral t":
		likelihood.Name = "noncentral_t"
		t := likelihoodObj.Get("parameters").Get("t").Float()
		df := likelihoodObj.Get("parameters").Get("df").Float()
		likelihood.Params = []float64{t, df}
	case "noncentral d":
		likelihood.Name = "noncentral_d"
		d := likelihoodObj.Get("parameters").Get("d").Float()
		n := likelihoodObj.Get("parameters").Get("n").Float()
		likelihood.Params = []float64{d, n}
	case "noncentral d2":
		likelihood.Name = "noncentral_d2"
		d := likelihoodObj.Get("parameters").Get("d").Float()
		n1 := likelihoodObj.Get("parameters").Get("n1").Float()
		n2 := likelihoodObj.Get("parameters").Get("n2").Float()
		likelihood.Params = []float64{d, n1, n2}
	default:
		print("nothing to do")
		return likelihood, nil

	}

	if err != nil {
		print("missing likelihoodDef")
		return likelihood, errors.New("missing likelihoodDef")
	}
	return likelihood, nil
}

// parse_prior
func parsePrior(args []js.Value, priorType string, likelihoodName string) (bayesfactor.PriorDefinition, error) {

	var nullprior bayesfactor.PriorDefinition // null prior
	nullpriorObj, _ := getParam(args[0], priorType)
	nullPriorName := nullpriorObj.Get("distribution").String()

	switch nullPriorName {
	case "normal":
		nullprior.Name = "normal"
		mean := nullpriorObj.Get("parameters").Get("mean").Float()
		sd := nullpriorObj.Get("parameters").Get("sd").Float()
		min := getMinMax(nullpriorObj, "min", likelihoodName)
		max := getMinMax(nullpriorObj, "max", likelihoodName)
		nullprior.Params = []float64{mean, sd, min, max}
	case "student t":
		nullprior.Name = "student_t"
		mean := nullpriorObj.Get("parameters").Get("mean").Float()
		sd := nullpriorObj.Get("parameters").Get("sd").Float()
		df := nullpriorObj.Get("parameters").Get("df").Float()
		min := getMinMax(nullpriorObj, "min", likelihoodName)
		max := getMinMax(nullpriorObj, "max", likelihoodName)
		nullprior.Params = []float64{mean, sd, df, min, max}
	case "beta":
		nullprior.Name = "beta"
		alpha := nullpriorObj.Get("parameters").Get("alpha").Float()
		beta := nullpriorObj.Get("parameters").Get("beta").Float()
		nullprior.Params = []float64{alpha, beta}
	case "cauchy":
		nullprior.Name = "cauchy"
		location := nullpriorObj.Get("parameters").Get("location").Float()
		scale := nullpriorObj.Get("parameters").Get("scale").Float()
		min := getMinMax(nullpriorObj, "min", likelihoodName)
		max := getMinMax(nullpriorObj, "max", likelihoodName)
		nullprior.Params = []float64{location, scale, min, max}
	case "uniform":
		nullprior.Name = "uniform"
		alpha := nullpriorObj.Get("parameters").Get("minimum").Float()
		beta := nullpriorObj.Get("parameters").Get("maximum").Float()
		nullprior.Params = []float64{alpha, beta}
	case "point":
		nullprior.Name = "point"
		point := nullpriorObj.Get("parameters").Get("point").Float()
		nullprior.Params = []float64{point}
	default:
		print("nothing to do")
		return nullprior, nil
	}

	return nullprior, nil

}

func bfWrapper(this js.Value, args []js.Value) interface{} {

	likelihood, _ := parseLikelihood(args)
	altprior, _ := parsePrior(args, "altpriorDef", likelihood.Name)
	nullprior, _ := parsePrior(args, "nullpriorDef", likelihood.Name)

	bf, err := bf(likelihood, altprior, nullprior)
	if err != nil {
		return nil
	}
	return bf
}
