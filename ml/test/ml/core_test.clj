(ns ml.core-test
  (:require [clojure.test :refer :all]
            [ml.core :refer :all]))

(deftest derivative-test
  (testing "FIXME, I fail."
    (is (= (derivative x (+ x 3)) 1))
    (is (= (derivative x (* x 3)) 3))
    (is (= (derivative x (* x 3 4)) 12))
    (is (= (derivative x (* x 3 y)) (* 3 'y)))
    ))
