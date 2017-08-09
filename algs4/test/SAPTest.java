import edu.princeton.cs.algs4.Digraph;
import edu.princeton.cs.algs4.In;

import java.util.Arrays;

import static org.junit.Assert.*;

/**
 * Created by cperkins on 11/7/15.
 */
public class SAPTest {

    SAP sap1;
    SAP sap2;
    Digraph dg1;
    Digraph dg2;

    @org.junit.Before
    public void setUp() throws Exception {
        dg1 = new Digraph(new In("digraph1.txt"));
        sap1 = new SAP(dg1);
        dg2 = new Digraph((new In("digraph2.txt")));
        sap2 = new SAP(dg2);
    }

    @org.junit.After
    public void tearDown() throws Exception {

    }

    @org.junit.Test
    public void testLength() throws Exception {
        assertEquals(4, sap1.length(3, 11));
        assertEquals(3, sap1.length(9, 12));
        assertEquals(4, sap1.length(7, 2));
        assertEquals(-1, sap1.length(1, 6));
        assertEquals(1, sap1.length(1, 0));

        assertEquals(2, sap2.length(1, 5));
    }

    @org.junit.Test
    public void testAncestor() throws Exception {
        assertEquals(1, sap1.ancestor(3, 11));
        assertEquals(5, sap1.ancestor(9, 12));
        assertEquals(0, sap1.ancestor(7, 2));
        assertEquals(-1, sap1.ancestor(1, 6));
        assertEquals(0, sap1.ancestor(1, 0));

        assertEquals(0, sap2.ancestor(1, 5));
    }

    @org.junit.Test
    public void testLengthSetToSet() throws Exception {
        assertEquals(3, sap1.length(Arrays.asList(new Integer[]{3, 8}), Arrays.asList(new Integer[]{10,12})));
    }

}