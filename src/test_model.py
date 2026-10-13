import unittest
import numpy as np
import torch
from train import reflected_roi,paired_loss
import model as b


class FlipTests(unittest.TestCase):
    def test_both_channels_reflect_without_mutating(self):
        x=torch.zeros(2,2,4,6);x[0,0,1,0]=.7;x[0,1,1,0]=1;x[1,0,2,1]=.3;x[1,1,2,1]=1
        before=x.clone();y=reflected_roi(x)
        self.assertEqual(float(y[0,0,1,5]),float(x[0,0,1,0]))
        self.assertEqual(float(y[0,1,1,5]),1)
        self.assertEqual(float(y[1,1,2,4]),1)
        torch.testing.assert_close(reflected_roi(y),x);torch.testing.assert_close(x,before)

    def test_paired_gradient_is_orientation_invariant(self):
        torch.manual_seed(2)
        class Tiny(torch.nn.Module):
            def __init__(self):
                super().__init__();self.linear=torch.nn.Linear(2*4*6+3,2)
            def forward(self,x,g):return self.linear(torch.cat([x.flatten(1),g],1))
        m=Tiny();x=torch.rand(4,2,4,6);g=torch.rand(4,3);q=torch.tensor([.8,.65,.2,0.]);binary=torch.tensor([1,1,-1,0]);group=torch.tensor([0,0,0,-1])
        a=paired_loss(m,x,g,q,binary,group);a.backward();grad=m.linear.weight.grad.clone();m.zero_grad()
        h=paired_loss(m,reflected_roi(x),g,q,binary,group);h.backward()
        torch.testing.assert_close(a,h);torch.testing.assert_close(m.linear.weight.grad,grad)
        self.assertGreater(float(grad.abs().sum()),0)

    def test_reflecting_candidate_and_gt_preserves_targets(self):
        width=64;gt=np.array([[12.,15.,25.,31.],[36.,20.,45.,29.]])
        pred=np.array([[12.,15.,25.,31.],[10.,15.,21.,31.],[1.,1.,5.,5.]])
        def flip(boxes):
            q=boxes.copy();q[:,0]=width-boxes[:,2];q[:,2]=width-boxes[:,0];return q
        original=np.array([b.iou(p,gt) for p in pred]);reflected=np.array([b.iou(p,flip(gt)) for p in flip(pred)])
        np.testing.assert_allclose(original,reflected,atol=1e-12)
        np.testing.assert_array_equal(original.max(1)>=.5,reflected.max(1)>=.5)
        np.testing.assert_array_equal(original.max(1)<=.1,reflected.max(1)<=.1)


if __name__=='__main__':unittest.main()
