import edu.princeton.cs.algs4.BreadthFirstDirectedPaths;
import edu.princeton.cs.algs4.Digraph;

import java.util.Arrays;

public class SAP {

    private final Digraph dg;

    // constructor takes a digraph (not necessarily a DAG)
    public SAP(Digraph G) {
        dg = G;
    }

    private static class IntTuple2 {
        public Integer _1;
        public Integer _2;

        public IntTuple2(Integer a, Integer b) {
            _1 = a;
            _2 = b;
        }
    }

    private IntTuple2 calcLengthAndAncestor(int v, int w) {
        return calcLengthAndAncestor(Arrays.asList(new Integer[]{v}), Arrays.asList(new Integer[]{w}));
    }

    private IntTuple2 calcLengthAndAncestor(Iterable<Integer> vs, Iterable<Integer> ws) {
        BreadthFirstDirectedPaths vPaths = new BreadthFirstDirectedPaths(dg, vs);
        BreadthFirstDirectedPaths wPaths = new BreadthFirstDirectedPaths(dg, ws);
        int min = -1;
        int ancestor = -1;
        for (int i = 0; i < dg.V(); ++i) {
            if (vPaths.hasPathTo(i) && wPaths.hasPathTo(i)) {
                int d = vPaths.distTo(i) + wPaths.distTo(i);
                if (d < min || min == -1) {
                    min = d;
                    ancestor = i;
                }
            }
        }
        return new IntTuple2(min, ancestor);
    }

    // length of shortest ancestral path between v and w; -1 if no such path
    public int length(int v, int w) {
        return calcLengthAndAncestor(v, w)._1;
    }

    // a common ancestor of v and w that participates in a shortest ancestral path; -1 if no such path
    public int ancestor(int v, int w) {
        return calcLengthAndAncestor(v, w)._2;
    }

    // length of shortest ancestral path between any vertex in v and any vertex in w; -1 if no such path
    public int length(Iterable<Integer> v, Iterable<Integer> w) {
        return calcLengthAndAncestor(v, w)._1;
    }

    // a common ancestor that participates in shortest ancestral path; -1 if no such path
    public int ancestor(Iterable<Integer> v, Iterable<Integer> w) {
        return calcLengthAndAncestor(v, w)._2;
    }

    // do unit testing of this class
    public static void main(String[] args) {

    }
}

