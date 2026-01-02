/**
 * カスタムマッチャー
 * Jest のアサーションを拡張
 */

/**
 * 数値が範囲内にあるかチェック
 */
expect.extend({
  toBeWithinRange(received: number, floor: number, ceiling: number) {
    const pass = received >= floor && received <= ceiling;
    if (pass) {
      return {
        message: () =>
          `expected ${received} not to be within range ${floor} - ${ceiling}`,
        pass: true,
      };
    } else {
      return {
        message: () =>
          `expected ${received} to be within range ${floor} - ${ceiling}`,
        pass: false,
      };
    }
  },
});

/**
 * 配列に重複がないかチェック
 */
expect.extend({
  toHaveNoDuplicates(received: any[]) {
    const uniqueSet = new Set(received);
    const pass = uniqueSet.size === received.length;

    if (pass) {
      return {
        message: () => `expected array to have duplicates`,
        pass: true,
      };
    } else {
      const duplicates = received.filter(
        (item, index) => received.indexOf(item) !== index
      );
      return {
        message: () =>
          `expected array to have no duplicates, but found: ${JSON.stringify(
            duplicates
          )}`,
        pass: false,
      };
    }
  },
});

/**
 * オブジェクトが特定のキーを持つかチェック
 */
expect.extend({
  toHaveKeys(received: object, expectedKeys: string[]) {
    const actualKeys = Object.keys(received);
    const missingKeys = expectedKeys.filter(
      (key) => !actualKeys.includes(key)
    );
    const pass = missingKeys.length === 0;

    if (pass) {
      return {
        message: () =>
          `expected object not to have keys: ${expectedKeys.join(', ')}`,
        pass: true,
      };
    } else {
      return {
        message: () =>
          `expected object to have keys: ${expectedKeys.join(
            ', '
          )}, but missing: ${missingKeys.join(', ')}`,
        pass: false,
      };
    }
  },
});

/**
 * 配列が昇順にソートされているかチェック
 */
expect.extend({
  toBeSortedAscending(received: any[]) {
    const sorted = [...received].sort((a, b) => a - b);
    const pass = JSON.stringify(received) === JSON.stringify(sorted);

    if (pass) {
      return {
        message: () => `expected array not to be sorted in ascending order`,
        pass: true,
      };
    } else {
      return {
        message: () =>
          `expected array to be sorted in ascending order, but got: ${JSON.stringify(
            received
          )}`,
        pass: false,
      };
    }
  },
});

/**
 * 配列が降順にソートされているかチェック
 */
expect.extend({
  toBeSortedDescending(received: any[]) {
    const sorted = [...received].sort((a, b) => b - a);
    const pass = JSON.stringify(received) === JSON.stringify(sorted);

    if (pass) {
      return {
        message: () => `expected array not to be sorted in descending order`,
        pass: true,
      };
    } else {
      return {
        message: () =>
          `expected array to be sorted in descending order, but got: ${JSON.stringify(
            received
          )}`,
        pass: false,
      };
    }
  },
});

/**
 * 文字列がメールアドレス形式かチェック
 */
expect.extend({
  toBeValidEmail(received: string) {
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    const pass = emailRegex.test(received);

    if (pass) {
      return {
        message: () => `expected ${received} not to be a valid email`,
        pass: true,
      };
    } else {
      return {
        message: () => `expected ${received} to be a valid email`,
        pass: false,
      };
    }
  },
});

/**
 * 文字列がURL形式かチェック
 */
expect.extend({
  toBeValidUrl(received: string) {
    try {
      new URL(received);
      return {
        message: () => `expected ${received} not to be a valid URL`,
        pass: true,
      };
    } catch {
      return {
        message: () => `expected ${received} to be a valid URL`,
        pass: false,
      };
    }
  },
});

/**
 * 日付が過去かチェック
 */
expect.extend({
  toBePastDate(received: Date | string) {
    const date = new Date(received);
    const now = new Date();
    const pass = date < now;

    if (pass) {
      return {
        message: () => `expected ${received} not to be a past date`,
        pass: true,
      };
    } else {
      return {
        message: () => `expected ${received} to be a past date`,
        pass: false,
      };
    }
  },
});

/**
 * 日付が未来かチェック
 */
expect.extend({
  toBeFutureDate(received: Date | string) {
    const date = new Date(received);
    const now = new Date();
    const pass = date > now;

    if (pass) {
      return {
        message: () => `expected ${received} not to be a future date`,
        pass: true,
      };
    } else {
      return {
        message: () => `expected ${received} to be a future date`,
        pass: false,
      };
    }
  },
});

/**
 * オブジェクトが空かチェック
 */
expect.extend({
  toBeEmptyObject(received: object) {
    const pass = Object.keys(received).length === 0;

    if (pass) {
      return {
        message: () => `expected object not to be empty`,
        pass: true,
      };
    } else {
      return {
        message: () =>
          `expected object to be empty, but has keys: ${Object.keys(
            received
          ).join(', ')}`,
        pass: false,
      };
    }
  },
});

/**
 * Promise が resolve するかチェック（タイムアウト付き）
 */
expect.extend({
  async toResolveWithin(
    received: Promise<any>,
    timeout: number
  ) {
    try {
      await Promise.race([
        received,
        new Promise((_, reject) =>
          setTimeout(() => reject(new Error('Timeout')), timeout)
        ),
      ]);
      return {
        message: () => `expected promise not to resolve within ${timeout}ms`,
        pass: true,
      };
    } catch (error) {
      return {
        message: () => `expected promise to resolve within ${timeout}ms`,
        pass: false,
      };
    }
  },
});

// TypeScript 型定義
declare global {
  namespace jest {
    interface Matchers<R> {
      toBeWithinRange(floor: number, ceiling: number): R;
      toHaveNoDuplicates(): R;
      toHaveKeys(keys: string[]): R;
      toBeSortedAscending(): R;
      toBeSortedDescending(): R;
      toBeValidEmail(): R;
      toBeValidUrl(): R;
      toBePastDate(): R;
      toBeFutureDate(): R;
      toBeEmptyObject(): R;
      toResolveWithin(timeout: number): Promise<R>;
    }
  }
}

export {};
