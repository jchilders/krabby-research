"""HalClient observation SUB socket uses latest-only ZMQ options."""

import zmq

from hal.client.client import HalClient
from hal.client.config import HalClientConfig
from hal.server import HalServerBase, HalServerConfig


def test_hal_client_observation_socket_rcvhwm_and_conflate():
    server_config = HalServerConfig(
        observation_bind="inproc://test_sub_opts_obs",
        command_bind="inproc://test_sub_opts_cmd",
    )
    server = HalServerBase(server_config)
    server.initialize()

    client_config = HalClientConfig(
        observation_endpoint="inproc://test_sub_opts_obs",
        command_endpoint="inproc://test_sub_opts_cmd",
    )
    client = HalClient(client_config, context=server.get_transport_context())
    client.initialize()

    assert client.observation_socket is not None
    assert client.observation_socket.getsockopt(zmq.RCVHWM) == 1
    assert client.observation_socket.getsockopt(zmq.CONFLATE) == 1

    client.close()
    server.close()
