/*!
 * activation-functions - v2.0.0 - 2020
 * https://github.com/howion/activation-functions
 *
 * Copyright (c) 2018 howion
 * Licensed under the MIT license */

/**
 * Identity function: x -> x
 * @param {number} $x
 */
export const Identity = ($x: number): number => {
    return $x;
};

/**
 * Inverse function: x -> 1-x
 * @param {number} $x
 */
export const Inverse = ($x: number): number => {
    return (1 - $x);
};

/**
 * BinaryStep function: x -> (x<0 ? 0 : 1)
 * @param {number} $x
 */
export const BinaryStep = ($x: number): number => {
    return (($x < 0) ? 0 : 1);
};

/**
 * Bipolar function: x -> (x>0 ? 1 : 0)
 * @param {number} $x
 */
export const Bipolar = ($x: number): number => {
    return (($x > 0) ? 1 : -1);
};

/**
 * Logistic, Sigmoid, or SoftStep function: x -> (1/(1+e^-x))
 * @param {number} $x
 */
export const Logistic = ($x: number): number => {
    return 1 / (1 + Math.exp(-$x));
};

export const Sigmoid = Logistic;
export const SoftStep = Logistic;

/**
 * BipolarSigmoid function: x -> (x>0 ? 1 : 0)
 * @param {number} $x
 */
export const BipolarSigmoid = ($x: number): number => {
    return (2 * Sigmoid($x) - 1);
};

/**
 * Same as Math.tanh
 * @param {number} $x (in radians)
 */
export const Tanh = ($x: number): number => {
    return Math.tanh($x);
};

/**
 * HardTanh function: x-> max(-1, min(1, x))
 * @param {number} $x (in radians)
 */
export const HardTanh = ($x: number): number => {
    return Math.max(-1, Math.min(1, $x));
};

/**
 * Same as Math.atan
 * @param {number} $x (in radians)
 */
export const ArcTan = ($x: number): number => {
    return Math.atan($x);
};

/**
 * ElliotSig, or SoftSign function: x -> (x/(1+|x|))
 * @param {number} $x
 */
export const ElliotSig = ($x: number): number => {
    return $x / (1 + Math.abs($x));
};

export const SoftSign = ElliotSig;

/**
 * Erf (Error) function: x -> erf(x)
 * @param {number} $x (in radians)
 */
export const Erf = ($x: number): number => {
    let a1 = 0.254829592;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;
    let p = 0.3275911;

    let sign = (($x < 0) ? -1 : 1);
    $x = Math.abs($x);

    let t = 1.0 / (1.0 + p * $x);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * Math.exp(-($x ** 2));

    return sign * y;
};

/**
 * Sinc function: x -> (sinx/x)
 * @param {number} $x (in radians)
 */
export const Sinc = ($x: number): number => {
    return (($x == 0) ? 1 : (Math.sin($x) / $x));
};

/**
 * Same as Math.sin
 * @param {number} $x (in radians)
 */
export const Sinusoid = ($x: number): number => {
    return Math.sin($x)
};

/**
 * Gaussian function: x -> e^(-x^2)
 * @param {number} $x
 */
export const Gaussian = ($x: number): number => {
    return Math.exp(-($x ** 2));
};

/**
 * ISRU (Inverse Square Root Unit) function: x -> x/sqrt(1+x^2)
 * @param {number} $x
 * @param $a
 */
export const ISRU = ($x: number, $a: number): number => {
    return ($x / Math.sqrt(1 + $a * ($x ** 2)));
};

/**
 * ReLU (Rectified linear unit) function: x -> max(0, x)
 * @param {number} $x
 */
export const ReLU = ($x: number): number => {
    return Math.max(0, $x);
};

/**
 * GELU (Gaussian error linear unit) function: x -> (x/2){1+erf[x/sqrt(2)]}
 *
 * See: https://arxiv.org/abs/1606.08415
 * @param {number} $x
 */
export const GELU = ($x: number): number => {
    return ($x / 2) * (1 + Erf($x / Math.SQRT2));
};

/**
 * PReLU (Parameteric rectified linear unit) function: x -> ((x<0) ? ax : x)
 *
 * See: https://arxiv.org/abs/1502.01852
 * @param {number} $x
 * @param $a
 */
export const PReLU = ($x: number, $a: number): number => {
    return (($x < 0) ? ($a * $x) : $x);
};

/**
 * ELU (Exponential Linear Unit) function: x -> ((x>0) ? x : (a*(e^x -1)))
 *
 * See: https://arxiv.org/abs/1511.07289
 * @param {number} $x
 * @param $a
 */
export const ELU = ($x: number, $a: number): number => {
    return (($x > 0) ? $x : ($a * Math.expm1($x)));
};

/**
 * SELU (Scaled Exponential Linear Unit) function: x -> 1.0507*ELU(1.67326, x)
 *
 * See: https://arxiv.org/abs/1706.02515
 * @param {number} $x
 */
export const SELU = ($x: number): number => {
    return 1.0507 * ELU($x, 1.67326);
};

/**
 * SoftPlus function: x -> ln(1+e^x)
 * @param {number} $x
 */
export const SoftPlus = ($x: number): number => {
    return Math.log(1 + Math.exp($x));
};

/**
 * Mish function: x -> x*tanh(SoftPlus(x)=ln(1+e^x))
 *
 * See: https://github.com/digantamisra98/Mish
 * @param {number} $x (in radians)
 */
export const Mish = ($x: number): number => {
    return $x * Math.tanh(SoftPlus($x));
};

/**
 * SQNL (Square nonlinearity) function: x -> ...
 * @param {number} $x
 */
export const SQNL = ($x: number): number => {
    if ($x > +2) {
        return +1;
    }
    if ($x < -2) {
        return -1;
    }
    if ($x < 0) {
        return $x + ($x ** 2) / 4;
    }
    /* -2=<x<0: */
    return $x + ($x ** 2) / 4;
};

/**
 * BentIdentity function: x -> [{sqrt(x^2 + 1) - 1}/2 + x]
 * @param {number} $x
 */
export const BentIdentity = ($x: number): number => {
    return ((Math.sqrt(($x ** 2) + 1) - 1) / 2) + $x;
};

/**
 * (Sigmoid linear unit) SiLU, or Swish1 function: x -> (x/(1+e^-x))
 * @param {number} $x
 */
export const SiLU = ($x: number): number => {
    return $x * Sigmoid($x);
};

export const Swish1 = SiLU
