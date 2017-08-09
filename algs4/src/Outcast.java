public class Outcast {

    private final WordNet wn;

    // constructor takes a WordNet object
    public Outcast(WordNet wordnet) {
        wn = wordnet;
    }

    // given an array of WordNet nouns, return an outcast
    public String outcast(String[] nouns) {
        int maxDist = 0;
        String outcast = null;
        for (String noun : nouns) {
            int d = 0;
            for (String other : nouns) {
                d += wn.distance(noun, other);
            }
            if (d > maxDist) {
                maxDist = d;
                outcast = noun;
            }
        }
        return outcast;
    }

    public static void main(String[] args) {

    }
}