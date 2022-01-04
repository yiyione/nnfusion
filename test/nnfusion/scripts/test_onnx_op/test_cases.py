import os
import shutil
import numpy as np
import tempfile
from torch import nn
import torch.onnx
from nnfusion.executor import Executor
from nnfusion.session import codegen, modify_nnfusion_rt, build
from nnfusion.data_format import cast_pytorch_tensor

class TestCase:
    """Simple test case"""

    def __init__(self, inputs, outputs):
        self.inputs = inputs
        self.outputs = outputs

    def run(self, executor):
        input_dict, output_dict = {}, {}
        for name, x in self.inputs.items():
            input_dict[name] = cast_pytorch_tensor(x)
        for name, x in self.outputs.items():
            output_dict[name] = cast_pytorch_tensor(torch.zeros_like(x))
        executor(input_dict, output_dict)

        result = True
        for name, expect in self.outputs.items():
            output = output_dict[name].reference
            if (torch.allclose(output, expect)):
                continue
            else:
                result = False
                print('Expect:')
                print(expect)
                print('Actual:')
                print(output)
        return result

class TestOp:
    """Simple test op"""

    def __init__(self, name, function):
        self.name = name
        self.function = function
        self.test_cases = []

    def add_test(self, inputs, outputs):
        self.test_cases.append(TestCase(inputs, outputs))

class Tester:

    def run_test(self, function, inputs, outputs):
        class TestModule(nn.Module):
            def forward(self, inputs):
                return function(inputs)

        # Generate onnx file
        model = TestModule()

        dir_ctx = tempfile.TemporaryDirectory(prefix="nnf_")
        workdir = dir_ctx.name

        onnx_dir = os.path.join(workdir, "test.onnx")
        if os.path.exists(onnx_dir):
            os.remove(onnx_dir)

        torch.onnx.export(
            model, list(inputs.values()), onnx_dir,
            export_params=True, opset_version=15,
            input_names = list(inputs.keys()), output_names = list(outputs.keys())
        )

        # Codegen and build
        rt_dir = os.path.join(workdir, "nnfusion_rt/cuda_codegen")
        if os.path.exists(rt_dir):
            shutil.rmtree(rt_dir)
        codegen(onnx_dir, "-f onnx -fhost_entry=1", workdir)

        modify_nnfusion_rt(rt_dir)
        build(rt_dir)

        # Run and check result
        test_case = TestCase(inputs, outputs)
        return test_case.run(Executor(rt_dir))

    def test_op(self, op_test_case):
        print(op_test_case.name + ":")
        failed = False
        for case in op_test_case.test_cases:
            if not self.run_test(op_test_case.function, case.inputs, case.outputs):
                failed = True
        if not failed:
            print("PASS")

onnx_ops = []

neg_op = TestOp("Neg", lambda inputs : torch.neg(inputs[0]))
x = torch.from_numpy(np.array([-4, 2]).astype(np.float32))
neg_op.add_test({ "input": x }, { "output": torch.neg(x) })
x = torch.from_numpy(np.random.randn(3, 4, 5).astype(np.float32))
neg_op.add_test({ "input": x }, { "output": torch.neg(x) })
onnx_ops.append(neg_op)

scatter_nd_op = TestOp("ScatterND", lambda inputs : torch.neg(inputs[0]))
# how to gen the onnx ?
adam_op = TestOp("ai.onnx.preview.training.Adam", lambda inputs : torch.neg(inputs[0]))
# ai.onnx.preview.training.Adam
optional_op = TestOp("Optional", lambda inputs : torch.neg(inputs[0]))
# no test case ?

tan_op = TestOp("Tan", lambda inputs : torch.tan(inputs[0]))
x = torch.from_numpy(np.array([-1, 0, 1]).astype(np.float32))
tan_op.add_test({ "input": x }, { "output": torch.tan(x)})
x = torch.from_numpy(np.random.randn(3, 4, 5).astype(np.float32))
tan_op.add_test({ "input": x }, { "output": torch.tan(x)})
onnx_ops.append(tan_op)

tester = Tester()
for op in onnx_ops:
    tester.test_op(op)
