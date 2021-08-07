package distributions

import (
	"math"
	"testing"

	"github.com/google/go-cmp/cmp"
)

func TestDistribution(t *testing.T) {

	// note: all test values are taken from R v4.0.2

	const tolerance = .0001
	opt := cmp.Comparer(func(x, y float64) bool {
		diff := math.Abs(x - y)
		mean := math.Abs(x+y) / 2.0
		return (diff / mean) < tolerance
	})

	// non-central t distribution

	got := Dt(10, 10, 10)
	want := 0.1601017

	if !cmp.Equal(got, want, opt) {
		t.Fatalf("got %v, wanted %v", got, want)
	}

	got = Dt(0.1, 12, .2)
	want = 0.3887798

	if !cmp.Equal(got, want, opt) {
		t.Fatalf("got %v, wanted %v", got, want)
	}

	got = Dt(9, 12, .4)
	want = 2.318372e-06

	if !cmp.Equal(got, want, opt) {
		t.Fatalf("got %v, wanted %v", got, want)
	}


	// cauchy distribution
	got = Dcauchy(0, 1, 1)
	want = 0.1591549

	if !cmp.Equal(got, want, opt) {
		t.Fatalf("got %v, wanted %v", got, want)
	}

	got = Dcauchy(0, .707, 1)
	want = 0.212228

	if !cmp.Equal(got, want, opt) {
		t.Fatalf("got %v, wanted %v", got, want)
	}

	// student t-distribution

	got = Scaled_shifted_t(.2, 1, 2, 12)
	want = 0.1792473

	if !cmp.Equal(got, want, opt) {
		t.Fatalf("got %v, wanted %v", got, want)
	}

	// beta distribution
	got = Dbeta(.2, 1, 2.5)
	want = 1.788854

	if !cmp.Equal(got, want, opt) {
		t.Fatalf("got %v, wanted %v", got, want)
	}

	got = Dbeta(.5, 2.4, 2.5)
	want = 1.676801

	if !cmp.Equal(got, want, opt) {
		t.Fatalf("got %v, wanted %v", got, want)
	}

	// binomial distribution
	got = Dbinom(3, 12, .6)
	want = 0.01245708

	if !cmp.Equal(got, want, opt) {
		t.Fatalf("got %v, wanted %v", got, want)
	}

	got = Dbinom(8, 11, .4)
	want = 0.02335703

	if !cmp.Equal(got, want, opt) {
		t.Fatalf("got %v, wanted %v", got, want)
	}

}
