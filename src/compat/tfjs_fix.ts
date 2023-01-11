// ugly code, see https://github.com/tensorflow/tfjs/issues/7259
export function nullWrapper<Wrapped, Result>(
    func: (v: Wrapped) => Result, value?: Wrapped
): Result | null {
    return value === undefined ? null : func(value);
}

export function undefinedWrapper<Wrapped, Result>(
    func: (v: Wrapped) => Result, value?: Wrapped
): Result | undefined {
    return value === undefined ? undefined : func(value);
}
