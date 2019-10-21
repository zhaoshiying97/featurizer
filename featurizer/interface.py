#!/usr/bin/env python3
# -*- coding: utf-8 -*-

class Functor(object):

    def forward(self, *args, **kwargs):
        raise NotImplementedError
    
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)