"""ZMQ PUB/SUB fan-out: two SUB sockets on one inproc PUB."""

import zmq


def test_inproc_pub_two_subscribers_both_receive_all_messages():
    """Two SUB sockets connect to the same inproc PUB; each should receive every send."""
    ctx = zmq.Context()
    topic = b"observation"
    endpoint = "inproc://task3_fanout_test"

    pub = ctx.socket(zmq.PUB)
    pub.bind(endpoint)

    sub_a = ctx.socket(zmq.SUB)
    sub_a.setsockopt(zmq.SUBSCRIBE, topic)
    sub_a.setsockopt(zmq.RCVTIMEO, 2000)
    sub_a.connect(endpoint)

    sub_b = ctx.socket(zmq.SUB)
    sub_b.setsockopt(zmq.SUBSCRIBE, topic)
    sub_b.setsockopt(zmq.RCVTIMEO, 2000)
    sub_b.connect(endpoint)

    import time

    # PUB/SUB: allow subscription to attach before first payload
    time.sleep(0.1)

    n = 5
    expected = [topic + f"payload{i}".encode() for i in range(n)]
    for i in range(n):
        pub.send(expected[i])
    time.sleep(0.05)

    received_a = [sub_a.recv() for _ in range(n)]
    received_b = [sub_b.recv() for _ in range(n)]

    pub.close(linger=0)
    sub_a.close(linger=0)
    sub_b.close(linger=0)
    ctx.term()

    assert received_a == expected
    assert received_b == expected
