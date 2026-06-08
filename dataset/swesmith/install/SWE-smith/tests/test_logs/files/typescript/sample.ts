// Simple TypeScript file for testing entity extraction

interface User {
  name: string;
  age: number;
}

class Calculator {
  private value: number;

  constructor(initial: number) {
    this.value = initial;
  }

  add(x: number): number {
    return this.value + x;
  }

  multiply(x: number): number {
    for (let i = 0; i < x; i++) {
      this.value *= 2;
    }
    return this.value;
  }
}

function greet(user: User): string {
  if (user.age > 18) {
    return `Hello, ${user.name}`;
  } else {
    return `Hi, ${user.name}`;
  }
}

const double = (n: number): number => n * 2;

var incrementCounter = function(): number {
  let counter = 0;
  return ++counter;
};
