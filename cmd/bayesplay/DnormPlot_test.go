package main


import (
	"math"
	"testing"
  // "fmt"
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

// func getElementByIndex(x interface{}, i int, element string) interface{} {
//   return x.([]interface{})[i].(map[string]interface{})[element]
// }

func TestMain(t *testing.T) {

  likelihoodPlotData := dnormPlot(0, 1)


  t.Log(getElementByIndex(likelihoodPlotData, 0, "x").(float64))
}
