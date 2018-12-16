(ns ml.core)

(defn derivative* [v expr]
  (let [dv (partial derivative* v)]
    (if (seq? expr)
     (let [[op & args] expr]
       (case op
         + (list* + (map dv args))
         * (list* * (remove #(= v %) args))))
     (if (= v expr)
       1
       0))))

(defmacro derivative [v expr]
  (derivative* v expr))

(defmacro defnd [name arglist & body]
  `(defn ~name ~arglist ~@body))

(defn infix* [expr]
  (if (seq? expr)
    (case (count expr)
      3 (let [[left op right] expr]
          (map infix* (list op left right)))
      (map infix* expr))
    expr))

(defmacro infix [expr]
  (infix* expr))

(defnd sigmoid [x]
  (infix (1 / (1 + (Math/exp (- x))))))
