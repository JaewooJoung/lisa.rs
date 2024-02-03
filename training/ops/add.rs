use std::ops::Add;

pub struct Tensor<T, N> {
    value: Vec<T>,
    graph: Graph<N>,
    grad: Vec<T>,
}

pub struct AddOp<T: Add<Output = T> + Copy, N> {
    input_a: Tensor<T, N>,
    input_b: Tensor<T, N>,
    output: Tensor<T, N>,
}

impl<T: Add<Output = T> + Copy, N> AddOp<T, N> {
    pub fn new(input_a: Tensor<T, N>, input_b: Tensor<T, N>) -> Self {
        assert_eq!(input_a.value.len(), input_b.value.len());
        assert_eq!(input_a.graph, input_b.graph);
        let graph = input_a.graph;
        let output = Tensor {
            value: vec![T::default(); input_a.value.len()],
            graph,
            grad: vec![T::default(); input_a.value.len()],
        };

        graph.operations.push(AddOp {
            input_a,
            input_b,
            output,
        });

        AddOp {
            input_a,
            input_b,
            output,
        }
    }

    pub fn forward(&mut self) {
        for i in 0..self.output.value.len() {
            self.output.value[i] = self.input_a.value[i] + self.input_b.value[i];
        }
    }

    pub fn backward(&mut self) {
        for i in 0..self.output.grad.len() {
            self.input_a.grad[i] += self.output.grad[i];
            self.input_b.grad[i] += self.output.grad[i];
        }
    }
}

impl<T: Add<Output = T> + Copy, N> Add for Tensor<T, N> {
    type Output = Self;

    fn add(self, other: Self) -> Self::Output {
        AddOp::new(self, other).output
    }
}
