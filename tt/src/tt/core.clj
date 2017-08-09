(ns tt.core
  (:require [clojure.edn :as edn]
            [clojure.java.io :as io]
            [clojure.pprint :as pprint :refer [print-table]]))

(def data-file "./zips.edn")

(defn city-id
  "Unique id identifying a city: [city state]"
  [record]
  [(:city record) (:state record)])

(defn load-data
  "Load the data and return as a sequence of maps with keyword keys."
  []
  (->> data-file
       slurp
       edn/read-string
       (map clojure.walk/keywordize-keys)
       (map #(assoc % :id (city-id %)))))

;; Compute population statistics: Min/Max/Average population of each _city_ (Notably, some cities span multiple zip codes).

(defn pop-by-city
  "Return map from city-id (vector of [city state]) to a map containing
  keys :min, :max, and :avg, containing the statistics of how populous that
  city's zip codes are."
  [data]
  (let [records-by-city (->> data
                             (sort-by :id)
                             (group-by :id))]
    (reduce-kv (fn [m id records]
                 (let [pops (map :pop records)
                       tot  (apply + pops)]
                   (assoc m id {:min (apply min pops)
                                :max (apply max pops)
                                :avg (/ tot (count pops))
                                :tot tot})))
               {}
               records-by-city)))


;; Find the northernmost and southernmost zip code and print their info.

(defn northernmost [data]
  (->> data
       (apply max-key #(second (:loc %)))))

(defn southernmost [data]
  (->> data
       (apply min-key #(second (:loc %)))))

;; Build an index of city name to zip codes (a mapping of strings to a
;; collection of zip codes).

(defn zips-by-city-name [data]
  (reduce (fn [m {:keys [city _id]}]
            (update m city #(conj (or % #{}) _id)))
          {}
          data))

;; Produce a list of states sorted by their population, together with the most
;; populous city in each state.

(defn states-by-pop [data]
  (let [pop-by-city  (pop-by-city data)
        pop-by-state (->> data
                          (map (fn [{:keys [state pop]}]
                                 {state pop}))
                          (apply merge-with +)
                          (sort-by second))]
    (for [[state state-pop] pop-by-state]
      {:state        :population   state-pop
       :largest-city (->> pop-by-city
                          (filter (fn [[[_ st] _]]
                                    (= st state)))
                          (apply max-key (fn [[_ {:keys [tot]}]] tot)))})))
