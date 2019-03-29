{-# LANGUAGE TopLevelKindSignatures #-}
{-# LANGUAGE PolyKinds, TypeFamilies #-}

module TLKS_024 where

import Data.Kind

data P (a :: k) = MkP

type C :: i -> Constraint
class C (p :: j) where
  type F :: j

f :: P k -> P (F :: k)
f _ = MkP
