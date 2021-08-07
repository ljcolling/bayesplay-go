package main
import (
.	"pkg/bayesfactor"
)


func dnormPlot(mean float64, sd float64) interface{} {

	min := mean - 4*sd // min range of plot
	max := mean + 4*sd // max range  of plot
	result := []interface{}{}

	likelihoodFunction := NormalLikelihood(mean, sd)

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
