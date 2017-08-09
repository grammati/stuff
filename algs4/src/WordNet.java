import edu.princeton.cs.algs4.Digraph;
import edu.princeton.cs.algs4.In;

import java.awt.*;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

public class WordNet {

    private Map<String, Set<Integer>> nounSynsetIds = new HashMap<String, Set<Integer>>();
    private Map<Integer, String> synsetsById = new HashMap<Integer, String>();
    private Digraph g;
    private final SAP sap;

    // constructor takes the name of the two input files
    public WordNet(String synsetsFile, String hypernymsFile) {
        In snIn = new In(synsetsFile);
        while (snIn.hasNextLine()) {
            String[] parts = snIn.readLine().split(",");
            int synsetId = Integer.parseInt(parts[0]);
            for (String noun : parts[1].split(" ")) {
                if (!nounSynsetIds.containsKey(noun)) {
                    nounSynsetIds.put(noun, new HashSet<Integer>());
                }
                nounSynsetIds.get(noun).add(synsetId);
            }
            synsetsById.put(synsetId, parts[1]);
        }

        g = new Digraph(nounSynsetIds.size());
        In hnIn = new In(hypernymsFile);
        while (hnIn.hasNextLine()) {
            String line = hnIn.readLine();
            String[] parts = line.split(",");
            int ssId = Integer.parseInt(parts[0]);
            for (int i = 1; i < parts.length; ++i) {
                int otherId = Integer.parseInt(parts[i]);
                g.addEdge(ssId, otherId);
            }
        }

        sap = new SAP(g);
    }

    // returns all WordNet nouns
    public Iterable<String> nouns() {
        return nounSynsetIds.keySet();
    }

    // is the word a WordNet noun?
    public boolean isNoun(String word) {
        return nounSynsetIds.containsKey(word);
    }

    // distance between nounA and nounB (defined below)
    public int distance(String nounA, String nounB) {
        int minDist = Integer.MAX_VALUE;
        for (Integer aSynSetId : nounSynsetIds.get(nounA)) {
            for (Integer bSynSetId : nounSynsetIds.get(nounB)) {
                int d = sap.length(aSynSetId, bSynSetId);
                if (d != -1 && d < minDist) {
                    minDist = d;
                }
            }
        }
        return minDist;
    }

    // a synset (second field of synsets.txt) that is the common ancestor of nounA and nounB
    // in a shortest ancestral path (defined below)
    public String sap(String nounA, String nounB) {
        int minDist = Integer.MAX_VALUE;
        int ancestor = -1;
        for (Integer aSynSetId : nounSynsetIds.get(nounA)) {
            for (Integer bSynSetId : nounSynsetIds.get(nounB)) {
                int d = sap.length(aSynSetId, bSynSetId);
                if (d != -1 && d < minDist) {
                    minDist = d;
                    ancestor = sap.ancestor(aSynSetId, bSynSetId);
                }
            }
        }
        return ancestor == -1 ? null : synsetsById.get(ancestor);
    }
}